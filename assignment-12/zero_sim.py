"""Data parallelism and ZeRO stages 1, 2, 3, written from scratch, on W virtual GPUs.

Every virtual GPU is one operating-system process started by torchrun; the processes talk through
torch.distributed with the gloo (CPU) backend. All gradient and weight traffic goes through the hand-written
ring collectives in dist_lib.py, so every byte sent is counted.

    stage "ddp"   every rank holds weights, gradients, fp32 master copy, Adam m and v for the whole model;
                  after backward the gradients are all-reduced (ring = reduce-scatter + all-gather).
    stage "zero1" optimizer state (master, m, v) sharded: rank r keeps 1/W of it. Gradients are still held in
                  full; after backward they are reduce-scattered, each rank updates its 1/W of the weights,
                  and the updated slices are all-gathered back into every rank's full weight copy.
    stage "zero2" + gradients sharded: as soon as backward finishes a unit, that unit's gradient is
                  reduce-scattered and the full copy dropped, so only 1/W of the gradient is ever kept.
    stage "zero3" + weights sharded: rank r stores only 1/W of the weights. Each unit's weights are
                  all-gathered just before its forward, dropped, all-gathered again before its backward,
                  dropped again. (Implemented with recomputation, see UnitFn.)

Weights of each unit live in one flat vector padded to a multiple of W; rank r owns the slice
[r*S, (r+1)*S) of every unit, S = padded size / W.

Usage (from the notebook):
    torchrun --standalone --nproc_per_node=32 zero_sim.py --stages zero2 --dtype fp32 --steps 8 --out z2.json
"""
import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from dist_lib import STATS, ring_all_gather_, ring_all_reduce_, ring_reduce_scatter_
from model import GPT, Config, upcast

DTYPES = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.bfloat16}
STAGE_ID = {"ddp": 0, "zero1": 1, "zero2": 2, "zero3": 3}
# Bytes per element in the lesson's mixed-precision recipe (bf16 weights and grads, fp32 optimizer state).
RECIPE_BYTES = {"weights": 2, "gradients": 2, "fp32 master": 4, "Adam m": 4, "Adam v": 4}


# ----------------------------------------------------------------------------------------------------------
# Measurement helpers
# ----------------------------------------------------------------------------------------------------------
class MemTracker:
    """Byte-exact ledger of the training-state tensors this rank holds.

    Categories: weights, gradients, fp32 master, Adam m, Adam v. Two ledgers are kept side by side:
      actual : numel * element_size, the bytes really allocated in this simulation
      recipe : numel * the lesson's bytes per element (2, 2, 4, 4, 4), so an fp32 run reads in lesson units
    Peaks are tracked for the sum over categories, with the per-category split at the moment of the peak.

    Example:
        mem = MemTracker()
        mem.alloc("Adam m", torch.zeros(1000))    # actual 4,000 B, recipe 4,000 B
        mem.alloc("weights", torch.zeros(1000))   # actual 4,000 B (fp32 here), recipe 2,000 B (bf16 in the lesson)
        mem.peak -> {"actual": 8000, "recipe": 6000}
    """

    def __init__(self):
        self.live = {"actual": defaultdict(int), "recipe": defaultdict(int)}
        self.peak = {"actual": 0, "recipe": 0}
        self.at_peak = {"actual": {}, "recipe": {}}

    def _add(self, cat, t, sign):
        self.live["actual"][cat] += sign * t.numel() * t.element_size()
        self.live["recipe"][cat] += sign * t.numel() * RECIPE_BYTES[cat]

    def alloc(self, cat, t):
        self._add(cat, t, +1)
        for k in ("actual", "recipe"):
            total = sum(self.live[k].values())
            if total > self.peak[k]:
                self.peak[k], self.at_peak[k] = total, dict(self.live[k])
        return t

    def free(self, cat, t):
        self._add(cat, t, -1)

    def transient(self, cat, t):
        """A tensor that exists only briefly (autograd's own gradient before we copy it): counts toward the peak."""
        self.alloc(cat, t)
        self.free(cat, t)

    def snapshot(self):
        return {k: dict(v) for k, v in self.live.items()}


class SavedTensorCounter:
    """Counts the bytes autograd keeps for the backward pass (the activations); each storage counted once.

    Storages listed in `exclude` (the weight buffers) are skipped, so only activations are counted.

    Example:
        x = torch.randn(4, 8, requires_grad=True)          # fp32
        with SavedTensorCounter() as c:
            y = (x * 2).sin()                             # sin saves its input (x*2): 4*8*4 = 128 bytes
        c.bytes -> 128                                     # x*2 by a Python number saves nothing
    """

    def __init__(self, exclude=()):
        self.exclude, self.seen, self.bytes = set(exclude), set(), 0

    def _pack(self, t):
        if not isinstance(t, nn.Parameter):
            st = t.untyped_storage()
            ptr = st.data_ptr()
            if ptr not in self.exclude and ptr not in self.seen:
                self.seen.add(ptr)
                self.bytes += st.nbytes()
        return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)


def rss_bytes():
    """Resident memory of this process as the operating system sees it (bytes)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return 0


# ----------------------------------------------------------------------------------------------------------
# The unit that forgets its internals: used by ZeRO-3 (and by the --ckpt option for the other stages)
# ----------------------------------------------------------------------------------------------------------
class UnitFn(torch.autograd.Function):
    """Run one unit without keeping anything from inside it for the backward pass.

    forward : (ZeRO-3: all-gather the unit's weights) -> run the unit under no_grad -> (ZeRO-3: drop them).
              Only the unit's input x is saved.
    backward: (ZeRO-3: all-gather the weights again) -> recompute the unit with autograd on -> backpropagate
              -> (ZeRO-3: reduce-scatter the weight gradients, drop weights and full gradients).

    Why recompute: if autograd kept the unit's graph, the graph would hold references to the gathered weights,
    and they could not be freed between forward and backward. FSDP avoids the recompute by freeing the storage
    under the graph and refilling it in a backward hook; recomputing is the simpler, fully robust way to get the
    same memory behaviour, at the price of one extra forward per step (activation checkpointing).

    `dummy` is a scalar that requires grad: it makes sure autograd calls backward() for the first unit too,
    whose real input (token ids) has no gradient.
    """

    @staticmethod
    def forward(ctx, x, dummy, eng, u):
        ctx.eng, ctx.u = eng, u
        ctx.save_for_backward(x)
        return eng.unit_forward_nograd(u, x)

    @staticmethod
    def backward(ctx, gout):
        (x,) = ctx.saved_tensors
        return ctx.eng.unit_backward(ctx.u, x, gout), None, None, None


# ----------------------------------------------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------------------------------------------
class Engine:
    """One rank's share of a data-parallel / ZeRO training run.

    Example (inside a torchrun process, W = 32):
        eng = Engine(GPT(Config()), "zero2", args, MemTracker())
        rec = eng.train_step(x, y, t=1, lr=3e-3)     # x, y: (B_rank, T) token ids
        rec["loss"] -> about 8.3 at step 1;  eng.mem.snapshot()["recipe"]["gradients"] -> 2 * 165,960 bytes
    """

    def __init__(self, model, stage, args, mem):
        self.stage, self.args, self.mem = STAGE_ID[stage], args, mem
        self.W, self.r = dist.get_world_size(), dist.get_rank()
        self.pdt = DTYPES[args.dtype]                                         # weights and gradients
        self.mdt = torch.float64 if args.dtype == "fp64" else torch.float32  # master copy and Adam state
        self.ckpt = args.ckpt or self.stage == 3
        self.units = list(model.units)
        self.dummy = torch.zeros((), requires_grad=True)
        self.recompute_max_bytes = 0
        self.grad_copy_fallback = False
        W, r = self.W, self.r
        self.U = []
        for mod in self.units:
            entries, off = [], 0
            for name, p in mod.named_parameters():
                entries.append(dict(name=name, shape=tuple(p.shape), n=p.numel(), off=off, p=p))
                off += p.numel()
            n_pad = math.ceil(off / W) * W
            S = n_pad // W
            mine = slice(r * S, (r + 1) * S)
            flat = torch.zeros(n_pad, dtype=self.mdt)                         # (n_pad,) the unit's initial weights
            for e in entries:
                flat[e["off"]:e["off"] + e["n"]].copy_(e["p"].detach().reshape(-1))
            U = dict(entries=entries, n=off, n_pad=n_pad, S=S, mine=mine)

            # Optimizer state: the whole unit (DDP) or only my slice of it (every ZeRO stage).
            master = flat if self.stage == 0 else flat[mine].clone()
            U["master"] = mem.alloc("fp32 master", master)
            U["m"] = mem.alloc("Adam m", torch.zeros_like(master))
            U["v"] = mem.alloc("Adam v", torch.zeros_like(master))

            # Weights used for compute: the full unit (stages 0-2) or only my slice (ZeRO-3).
            if self.stage <= 2:
                full = mem.alloc("weights", flat.to(self.pdt, copy=True))
                for e in entries:                                             # module weights become views of it
                    e["p"].data = full[e["off"]:e["off"] + e["n"]].view(e["shape"])
                U["full_param"] = full
            else:
                U["shard_param"] = mem.alloc("weights", flat[mine].to(self.pdt, copy=True))
                for e in entries:                                             # the module keeps no weights at all
                    e["p"].data = torch.empty(0, dtype=self.pdt)

            # Gradients: the full unit (DDP, ZeRO-1) or only my slice (ZeRO-2, ZeRO-3).
            if self.stage <= 1:
                g = mem.alloc("gradients", torch.zeros(n_pad, dtype=self.pdt))
                for e in entries:                                             # autograd accumulates straight into g
                    e["p"].grad = g[e["off"]:e["off"] + e["n"]].view(e["shape"])
                U["full_grad"] = g
            else:
                U["shard_grad"] = mem.alloc("gradients", torch.zeros(S, dtype=self.pdt))
                U["tmp_grad"], U["pending"] = None, len(entries)
            self.U.append(U)
            del flat

        if self.stage == 2:                                                   # reduce each unit as soon as it is done
            for u, U in enumerate(self.U):
                for e in U["entries"]:
                    e["p"].register_post_accumulate_grad_hook(self._make_zero2_hook(u, e))
        self.n_pad_total = sum(U["n_pad"] for U in self.U)
        self.n_total = sum(U["n"] for U in self.U)

    # ---- helpers -------------------------------------------------------------------------------------------
    def state_ptrs(self):
        """Storage addresses of the weight buffers, so the activation counter can ignore them."""
        return [U["full_param"].untyped_storage().data_ptr() for U in self.U if "full_param" in U]

    def gather_unit(self, u, op):
        """ZeRO-3: rebuild unit u's full flat weights from everyone's slices.  (S,) per rank -> (n_pad,)"""
        U = self.U[u]
        full = self.mem.alloc("weights", torch.empty(U["n_pad"], dtype=self.pdt))
        full[U["mine"]].copy_(U["shard_param"])
        ring_all_gather_(full, op)
        return full

    @staticmethod
    def views(U, full, requires_grad=False):
        """Name -> tensor view into the flat buffer, in the unit's own parameter shapes (no copies)."""
        d = {}
        for e in U["entries"]:
            t = full[e["off"]:e["off"] + e["n"]].view(e["shape"])
            d[e["name"]] = t.detach().requires_grad_() if requires_grad else t
        return d

    def reduce_scatter_unit(self, u, flat_grad):
        """Sum unit u's gradient over all ranks; keep only my slice (ZeRO-2 and ZeRO-3)."""
        mine = ring_reduce_scatter_(flat_grad, "reduce-scatter grads")
        self.U[u]["shard_grad"].add_(mine)

    def _make_zero2_hook(self, u, e):
        def hook(p):
            U = self.U[u]
            if U["tmp_grad"] is None:                                         # first gradient of this unit
                U["tmp_grad"] = self.mem.alloc("gradients", torch.zeros(U["n_pad"], dtype=self.pdt))
            self.mem.transient("gradients", p.grad)                           # autograd's copy, dropped below
            U["tmp_grad"][e["off"]:e["off"] + e["n"]].copy_(p.grad.reshape(-1))
            p.grad = None
            U["pending"] -= 1
            if U["pending"] == 0:                                             # whole unit done: send it, drop it
                self.reduce_scatter_unit(u, U["tmp_grad"])
                self.mem.free("gradients", U["tmp_grad"])
                U["tmp_grad"], U["pending"] = None, len(U["entries"])
        return hook

    # ---- forward / backward of one unit -----------------------------------------------------------------
    def unit_forward_nograd(self, u, x):
        mod = self.units[u]
        if self.stage == 3:
            full = self.gather_unit(u, "all-gather weights (forward)")
            out = functional_call(mod, self.views(self.U[u], full), (x,))
            self.mem.free("weights", full)
            del full
            return out
        return mod(x)

    def unit_backward(self, u, x, gout):
        mod, U = self.units[u], self.U[u]
        xin = x.detach().requires_grad_() if x.is_floating_point() else x
        if self.stage == 3:
            full = self.gather_unit(u, "all-gather weights (backward)")
            params = self.views(U, full, requires_grad=True)
            with SavedTensorCounter(self.state_ptrs() + [full.untyped_storage().data_ptr()]) as c:
                with torch.enable_grad():
                    out = functional_call(mod, params, (xin,))
            self.recompute_max_bytes = max(self.recompute_max_bytes, c.bytes)
            inputs = list(params.values()) + ([xin] if xin.requires_grad else [])
            grads = torch.autograd.grad(out, inputs, gout)
            pad = U["n_pad"] - U["n"]
            flat_g = torch.cat([g.reshape(-1) for g in grads[:len(params)]] + ([grads[0].new_zeros(pad)] if pad else []))
            self.mem.alloc("gradients", flat_g)
            del out, params
            self.mem.free("weights", full)
            del full
            self.reduce_scatter_unit(u, flat_g)
            self.mem.free("gradients", flat_g)
            return grads[-1] if xin.requires_grad else None
        with SavedTensorCounter(self.state_ptrs()) as c:
            with torch.enable_grad():
                out = mod(xin)
        self.recompute_max_bytes = max(self.recompute_max_bytes, c.bytes)
        torch.autograd.backward(out, gout)                                    # fills p.grad (stage 0/1) or hooks (2)
        return xin.grad if xin.requires_grad else None

    def forward(self, idx):
        """idx (B, T) -> logits (B, T, V)."""
        x = idx
        for u, mod in enumerate(self.units):
            x = UnitFn.apply(x, self.dummy, self, u) if self.ckpt else mod(x)
        return x

    # ---- one training step ----------------------------------------------------------------------------
    def zero_grad(self):
        for U in self.U:
            (U["full_grad"] if self.stage <= 1 else U["shard_grad"]).zero_()

    def _check_grad_views(self):
        """DDP/ZeRO-1: autograd should have accumulated into our flat buffer in place. If a PyTorch version
        allocates a fresh .grad instead, copy it into the buffer (correct, just one extra copy)."""
        for U in self.U:
            for e in U["entries"]:
                view = U["full_grad"][e["off"]:e["off"] + e["n"]].view(e["shape"])
                p = e["p"]
                if p.grad is not None and p.grad.data_ptr() != view.data_ptr():
                    self.grad_copy_fallback = True
                    view.copy_(p.grad)
                    p.grad = view

    def sync_grads(self):
        """After backward. DDP: all-reduce every unit's gradient. ZeRO-1: reduce-scatter it. ZeRO-2/3: done already."""
        if self.stage == 0 and self.args.bug != "no_average":
            for U in reversed(self.U):
                ring_all_reduce_(U["full_grad"], "all-reduce grads")
        elif self.stage == 1:
            for U in reversed(self.U):
                ring_reduce_scatter_(U["full_grad"], "reduce-scatter grads")

    def optimizer_step(self, t, lr):
        """Adam (bias-corrected, no weight decay) on the slice of the weights this rank owns.

        Returns (global gradient norm, number of weights this rank updated).
            m <- b1 m + (1 - b1) g          v <- b2 v + (1 - b2) g^2
            w <- w - lr / (1 - b1^t) * m / (sqrt(v / (1 - b2^t)) + eps)
        """
        a, W = self.args, self.W
        owned = []                                                            # summed gradients of what I update
        for U in self.U:
            if self.stage == 0:
                owned.append(U["full_grad"])
            elif self.stage == 1:
                owned.append(U["full_grad"][U["mine"]])
            else:
                owned.append(U["shard_grad"])
        scale = 1.0 if a.bug in ("no_average", "no_divide") else 1.0 / W     # sum over ranks -> mean
        g32 = [g.to(self.mdt) * scale for g in owned]                         # fp32 (or fp64) temporaries

        # Global gradient norm for clipping (and logging). DDP has every gradient locally; under ZeRO each rank
        # has only its slice, so the squared norms are summed over ranks: one tiny all-reduce.
        sq = torch.zeros(max(W, 1), dtype=torch.float64)
        sq[0] = sum(float((g.double() ** 2).sum()) for g in g32)
        if self.stage >= 1:
            ring_all_reduce_(sq, "all-reduce grad-norm")
        gnorm = float(sq.sum()) ** 0.5
        coef = min(1.0, a.clip / (gnorm + 1e-6)) if a.clip > 0 else 1.0

        b1, b2, eps = 0.9, 0.95, 1e-8
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        n_upd = 0
        for U, g in zip(self.U, g32):
            if coef != 1.0:
                g.mul_(coef)
            m, v, w = U["m"], U["v"], U["master"]
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            w.addcdiv_(m, (v / bc2).sqrt_().add_(eps), value=-lr / bc1)
            n_upd += w.numel()

        # Write the new weights into the copy used for compute.
        for U in self.U:
            if self.stage == 0:
                U["full_param"].copy_(U["master"])
            elif self.stage in (1, 2):
                U["full_param"][U["mine"]].copy_(U["master"])                # my slice is new, the rest is stale ...
                if a.bug != "skip_allgather":
                    ring_all_gather_(U["full_param"], "all-gather weights (after update)")   # ... until this
            else:
                U["shard_param"].copy_(U["master"])                            # ZeRO-3 keeps only the slice
        return gnorm, n_upd

    def train_step(self, x, y, t, lr):
        """Forward, backward, gradient sync, optimizer. Returns the loss and the time split of the step."""
        self.zero_grad()
        T0, C0 = time.perf_counter(), STATS.seconds
        with SavedTensorCounter(self.state_ptrs()) as act:
            logits = self.forward(x)                                          # (B, T, V)
            loss = F.cross_entropy(upcast(logits).reshape(-1, logits.shape[-1]), y.reshape(-1))
        T1, C1 = time.perf_counter(), STATS.seconds
        loss.backward()
        if self.stage <= 1:
            self._check_grad_views()
        T2, C2 = time.perf_counter(), STATS.seconds
        self.sync_grads()
        T3, C3 = time.perf_counter(), STATS.seconds
        gnorm, n_upd = self.optimizer_step(t, lr)
        T4, C4 = time.perf_counter(), STATS.seconds
        return dict(loss=loss.item(), grad_norm=gnorm, n_updated=n_upd, act_bytes=act.bytes,
                    fwd_compute=(T1 - T0) - (C1 - C0), fwd_comm=C1 - C0,
                    bwd_compute=(T2 - T1) - (C2 - C1), bwd_comm=C2 - C1,
                    sync_comm=C3 - C2, opt_compute=(T4 - T3) - (C4 - C3), opt_comm=C4 - C3, total=T4 - T0)

    # ---- measurement only (library collectives, not counted) ----------------------------------------------
    def full_master(self):
        """The whole model's master weights as one (n_total,) fp64 vector, gathered from the slices."""
        parts = []
        for U in self.U:
            if self.stage == 0:
                full = U["master"]
            else:
                full = torch.empty(U["n_pad"], dtype=U["master"].dtype)
                dist.all_gather(list(full.chunk(self.W)), U["master"].contiguous())
            parts.append(full[:U["n"]].double())
        return torch.cat(parts)

    def replica_max_diff(self):
        """Largest |w_rank - w_rank0| over all ranks and all weights of the compute copy (stages 0-2 only)."""
        if self.stage == 3:
            return None
        mine = torch.cat([U["full_param"][:U["n"]].double() for U in self.U])
        ref = mine.clone()
        dist.broadcast(ref, src=0)
        d = torch.tensor([float((mine - ref).abs().max())], dtype=torch.float64)
        dist.all_reduce(d, op=dist.ReduceOp.MAX)
        return float(d)


# ----------------------------------------------------------------------------------------------------------
# Data and the run
# ----------------------------------------------------------------------------------------------------------
class Data:
    """Random windows from the TinyStories token stream; same windows on every rank, each rank takes its slice.

    The global batch of step t is drawn from a generator seeded with (seed, t), so a run on 1 rank and a run on
    32 ranks see exactly the same 64 sequences at every step: rank r takes sequences [r*b, (r+1)*b), b = 64 / W.
    Example: Data(...).batch(1) on rank 3 of 32 -> x, y of shape (2, 128): sequences 6 and 7 of step 1's batch.
    """

    def __init__(self, path, seq, global_batch, seed, W, r):
        self.tok = np.load(path, mmap_mode="r")
        self.n_eval = int(0.05 * len(self.tok))
        self.n_train = len(self.tok) - self.n_eval
        self.seq, self.gb, self.seed, self.W, self.r = seq, global_batch, seed, W, r
        assert global_batch % W == 0, "the global batch must split evenly over the ranks"
        self.b = global_batch // W

    def _windows(self, starts):
        w = np.stack([np.asarray(self.tok[s:s + self.seq + 1], dtype=np.int64) for s in starts])
        w = torch.from_numpy(w)                                               # (b, T + 1)
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()

    def batch(self, t):
        rng = np.random.default_rng([self.seed, t])
        starts = rng.integers(0, self.n_train - self.seq - 1, size=self.gb)
        return self._windows(starts[self.r * self.b:(self.r + 1) * self.b])

    def eval_batch(self, n_windows=64):
        starts = self.n_train + np.arange(n_windows) * (self.seq + 1)
        k = n_windows // self.W
        return self._windows(starts[self.r * k:(self.r + 1) * k])


def lr_at(t, a):
    """Linear warmup over a.warmup steps, then cosine down to 10% of the peak at the last step."""
    if t <= a.warmup:
        return a.lr * t / a.warmup
    p = (t - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def avg_over_ranks(x):
    t = torch.tensor([float(x)], dtype=torch.float64)
    dist.all_reduce(t)
    return float(t) / dist.get_world_size()


def run_stage(stage, a, data, cfg):
    W, r = dist.get_world_size(), dist.get_rank()
    STATS.reset()
    gc.collect()
    rss0 = rss_bytes()
    torch.manual_seed(a.seed)
    model = GPT(cfg)                                                          # same initial weights on every rank
    mem = MemTracker()
    eng = Engine(model, stage, a, mem)
    mem_after_init = mem.snapshot()
    hist = []
    xe, ye = data.eval_batch() if a.eval_every else (None, None)
    for t in range(1, a.steps + 1):
        x, y = data.batch(t)
        dist.barrier()                                                        # line the ranks up: time = this step
        rec = eng.train_step(x, y, t, lr_at(t, a))
        rec["t"], rec["lr"] = t, lr_at(t, a)
        rec["loss_global"] = avg_over_ranks(rec["loss"])
        if a.drift:
            rec["replica_max_diff"] = eng.replica_max_diff()
        if a.eval_every and (t % a.eval_every == 0 or t == 1):
            saved = STATS.save()                                              # evaluation traffic is not training
            with torch.no_grad():
                le = F.cross_entropy(upcast(eng.forward(xe)).reshape(-1, cfg.vocab_size), ye.reshape(-1))
            STATS.restore(saved)
            rec["eval_loss"] = avg_over_ranks(le.item())
        hist.append(rec)
        if r == 0 and (t <= 2 or t == a.steps or (a.steps > 20 and t % max(1, a.steps // 10) == 0)):
            extra = f"  held-out loss {rec['eval_loss']:.4f}" if "eval_loss" in rec else ""
            print(f"  [{stage}] step {t:4d}  mean loss over ranks {rec['loss_global']:.4f}  "
                  f"step time {rec['total']:.3f} s{extra}", flush=True)
    res = dict(rank=r, mem_after_init=mem_after_init, mem_end=mem.snapshot(), mem_peak=mem.peak,
               mem_at_peak=mem.at_peak, comm=STATS.as_dict(), hist=hist,
               recompute_max_bytes=eng.recompute_max_bytes, grad_copy_fallback=eng.grad_copy_fallback,
               rss_growth=rss_bytes() - rss0)
    final_diff = eng.replica_max_diff()
    weights = eng.full_master() if a.save_weights else None
    all_res = [None] * W
    dist.all_gather_object(all_res, res)
    out = None
    if r == 0:
        out = dict(stage=stage, W=W, dtype=a.dtype, ckpt=eng.ckpt, bug=a.bug, clip=a.clip, steps=a.steps,
                   micro_batch=data.b, seq=a.seq, global_batch=a.global_batch, n_params=eng.n_total,
                   n_params_padded=eng.n_pad_total, unit_sizes=[U["n"] for U in eng.U],
                   unit_sizes_padded=[U["n_pad"] for U in eng.U], param_bytes=torch.empty(0, dtype=eng.pdt).element_size(),
                   replica_max_diff_end=final_diff, ranks=all_res)
        if weights is not None:
            path = a.save_weights.replace("{stage}", stage)
            torch.save(weights, path)
            out["weights_file"] = path
    del eng, model, mem
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="ddp", help="comma list of ddp, zero1, zero2, zero3")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--global_batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ckpt", action="store_true", help="activation checkpointing for stages ddp/zero1/zero2")
    ap.add_argument("--bug", default="none", choices=["none", "no_average", "skip_allgather", "no_divide"])
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--drift", action="store_true", help="measure replica divergence after every step")
    ap.add_argument("--save_weights", default="", help="path for the final weights, '{stage}' is replaced")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per process")
    ap.add_argument("--pin", action="store_true", help="pin each rank to its own CPU core (needs W <= cores)")
    ap.add_argument("--data", default="data/tokens.npy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dist.init_process_group("gloo")
    torch.set_num_threads(a.threads)
    W, r = dist.get_world_size(), dist.get_rank()
    pinned = None
    if a.pin and hasattr(os, "sched_setaffinity"):                            # Linux: one core per virtual GPU
        cores = sorted(os.sched_getaffinity(0))
        if W <= len(cores):
            pinned = cores[int(os.environ.get("LOCAL_RANK", r))]
            os.sched_setaffinity(0, {pinned})
    cfg = Config()
    data = Data(a.data, a.seq, a.global_batch, a.seed, W, r)
    if r == 0:
        print(f"world size W = {W} processes (virtual GPUs), {a.threads} CPU thread(s) each, dtype {a.dtype}, "
              f"micro-batch {data.b} x {a.seq} tokens per rank, global batch {a.global_batch}"
              + (f", each rank pinned to its own core (rank 0 -> core {pinned})" if pinned is not None else ""), flush=True)
    results = {}
    for stage in a.stages.split(","):
        t0 = time.time()
        out = run_stage(stage, a, data, cfg)
        if r == 0:
            results[stage] = out
            print(f"  [{stage}] finished {a.steps} steps in {time.time() - t0:.1f} s", flush=True)
        dist.barrier()
    if r == 0:
        with open(a.out, "w") as f:
            json.dump(results, f)
        print(f"results written to {a.out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
