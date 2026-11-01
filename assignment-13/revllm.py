"""A ~20M-parameter Llama-style LM whose residual stream can be wired as a reversible ODE integrator.

Architecture (one block = the usual pre-norm Llama block, written as an *increment* f(p)):

    token ids (B, T)
      -> Embedding (tied with the LM head)                 x  (B, T, D)   fp32 residual stream
      -> residual stack, one of four wirings (L blocks):
           euler       p^{l+1} = p^l + f_l(p^l)                          standard residual (not invertible)
           midpoint    p^{l+1} = p^{l-1} + 2h f_l(p^l)                   reversible, 2 states
           leapfrog    p^{l+1} = 2 p^l - p^{l-1} + h^2 f_l(p^l)          reversible, 2 states
           hamiltonian q' = q + Attn_l(LN(p)),  p' = p + MLP_l(LN(q'))   reversible, 2 streams
         with f_l(p) = a + MLP_l(LN2(p + a)),  a = Attn_l(LN1(p))   (Gal et al. 2025, eq. 5)
      -> RMSNorm -> logits = h @ E^T  (B, T, V)  -> chunked, checkpointed cross-entropy

Notation in docstrings and shape comments:
    B = sequences per batch   T = sequence length   D = d_model   H = heads   hd = D / H
    F = SwiGLU hidden width   V = vocab size        L = number of blocks       N = B * T tokens

Starting states of the two-state schemes ("start"):
    ghost : p^{-1} = p^0 = x.  Every block uses the reversible rule; the first step is an Euler step
            (midpoint: p^1 = x + 2h f_0(x);  leapfrog: p^1 = x + h^2 f_0(x), i.e. zero initial velocity).
    euler : p^0 = x, p^1 = x + c f_0(x) with c = h (midpoint) or h^2/2 (leapfrog), then the rule from block 1.
The output read by the head is p^L (hamiltonian: (p^L + q^L) / 2).

Stream precision. The residual stream(s) can be kept in fp32, bf16 or fp64 ("stream"). Each block casts its
input to its own parameter dtype (fp32) before computing, so an fp64 stream changes only the stream arithmetic.
That matters for reversibility: the backward pass re-evaluates every block on a *rebuilt* state, and the block's
bf16 matrix products round their inputs to 8 bits. With an fp32 stream the rebuilt state differs from the forward
one by fp32 rounding, which occasionally flips a bf16 rounding and changes the block's output by a bf16 step;
going down the stack those differences are amplified. With an fp64 stream the rebuilt state, cast to fp32, is
(almost always) bit-identical to the forward one, so the recompute repeats the forward exactly.

reversible=True runs the stack inside RevStack (a torch.autograd.Function): the forward pass keeps only
the last two states; the backward pass rebuilds every earlier state from them, one block at a time, and
back-propagates through that block. reversible=False runs the very same arithmetic with ordinary autograd
(every activation stored), which is what the gradient checks compare against.
"""
import gc
import math
import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

SCHEMES = ("euler", "midpoint", "leapfrog", "hamiltonian")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}


@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 10          # L
    n_head: int = 6            # H   (hd = 384 / 6 = 64)
    d_model: int = 384         # D
    d_ff: int = 1024           # F   (8/3 * D rounded up to a multiple of 64)
    max_seq: int = 512         # T the RoPE tables are built for
    rope_base: float = 10000.0
    scheme: str = "euler"      # one of SCHEMES
    h: float = 0.5             # step size of midpoint / leapfrog
    start: str = "ghost"       # "ghost" or "euler" (see module docstring)
    reversible: bool = False   # True: RevStack (activations rebuilt), False: ordinary autograd
    dropout: float = 0.0       # must be 0 for exact reversibility (the counter-example sets it > 0)
    stream: str = "fp32"       # dtype of the residual stream(s): "fp32", "bf16" or "fp64"
    ce_chunk: int = 8192       # tokens per cross-entropy chunk; 0 = one un-chunked, un-checkpointed CE
    emb_mult: float = 1.0      # x = emb(idx) * emb_mult: scales the residual stream's starting size (head unaffected)

    def label(self):
        """Short run name, e.g. Config(scheme="midpoint", h=0.5, reversible=True).label() -> 'midpoint h=0.5 rev'"""
        s = self.scheme if self.scheme in ("euler", "hamiltonian") else f"{self.scheme} h={self.h:g}"
        if self.scheme in ("midpoint", "leapfrog") and self.start != "ghost":
            s += f" start={self.start}"
        s += " rev" if self.reversible else ""
        s += f" {self.stream}" if self.stream != "fp32" else ""
        return s + (f" emb x{self.emb_mult:g}" if self.emb_mult != 1.0 else "")


# ----------------------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * w, computed in the input's precision (fp32 stream -> fp32).

    Example: RMSNorm(4)(torch.tensor([[3., -3., 3., -3.]])) -> tensor([[1., -1., 1., -1.]])
    """

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):                                                   # (..., D) -> (..., D)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(hd, T, base):
    """cos, sin tables for rotary position embedding, each (T, hd/2).

    Example: rope_tables(4, 2, 10000.)[0] -> tensor([[1., 1.], [0.5403, 0.9999]])   (cos of angle t * base^(-2i/hd))
    """
    inv = base ** (-torch.arange(0, hd, 2).float() / hd)                      # (hd/2,)
    ang = torch.arange(T).float()[:, None] * inv[None]                        # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate each (even, odd) pair of features of x (B, H, T, hd) by its position's angle. Norm-preserving.

    Example: one head, hd=2, position 1, angle 1 rad: (1, 0) -> (cos 1, sin 1) = (0.5403, 0.8415)
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]                                       # (B, H, T, hd/2) each
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)     # (B, H, T, hd/2, 2)
    return out.flatten(-2)                                                    # (B, H, T, hd)


class Attention(nn.Module):
    """Causal multi-head self-attention with RoPE. (B, T, D) -> (B, T, D). No biases.

    Example: Attention(Config())(torch.randn(2, 16, 384)).shape -> torch.Size([2, 16, 384])
    """

    def __init__(self, cfg):
        super().__init__()
        self.H, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        cos, sin = rope_tables(self.hd, cfg.max_seq, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)   # 3 x (B, H, T, hd)
        cos, sin = self.cos[:T].to(q.dtype), self.sin[:T].to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            # Pin the kernel, so the reversible backward's recompute uses the same one as the forward pass.
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)             # (B, H, T, hd)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.proj(y.transpose(1, 2).reshape(B, T, D))                               # (B, T, D)
        return F.dropout(y, self.dropout, self.training)


class SwiGLU(nn.Module):
    """Gated feed-forward: w2( silu(w1 x) * w3 x ). (B, T, D) -> (B, T, D), hidden width F.

    Example: SwiGLU(Config())(torch.randn(2, 16, 384)).shape -> torch.Size([2, 16, 384])
    """

    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        return F.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)), self.dropout, self.training)


class Block(nn.Module):
    """One transformer block, exposed as increments instead of 'x + ...' so every wiring can reuse it.

    delta(p)      = a + MLP(LN2(p + a)),  a = Attn(LN1(p))      the whole block's update f(p)
    attn_delta(p) = Attn(LN1(p))                                  the hamiltonian scheme uses the two halves
    mlp_delta(q)  = MLP(LN2(q))

    Example: b = Block(Config()); p = torch.randn(1, 8, 384); torch.allclose(p + b.delta(p),
             (lambda y: y + b.mlp(b.norm2(y)))(p + b.attn(b.norm1(p))))  -> True   (= the standard pre-norm block)
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def _in(self, p):
        """Cast the stream to the block's parameter dtype: fp64 or bf16 stream -> fp32 compute; no-op for fp32."""
        return p.to(self.norm1.weight.dtype)

    def attn_delta(self, p):
        return self.attn(self.norm1(self._in(p)))

    def mlp_delta(self, q):
        return self.mlp(self.norm2(self._in(q)))

    def delta(self, p):
        p = self._in(p)
        a = self.attn(self.norm1(p))                    # (B, T, D), bf16 under autocast
        return a + self.mlp(self.norm2(p + a))          # (B, T, D)

    def attn_params(self):
        return list(self.norm1.parameters()) + list(self.attn.parameters())

    def mlp_params(self):
        return list(self.norm2.parameters()) + list(self.mlp.parameters())


# ------------------------------------------------------------------------- the four wirings, forward
def _coeffs(cfg):
    """(alpha, beta, gamma) of the two-state rule  c = alpha*b + beta*a + gamma*f(b), and the Euler-start factor.

    Example: _coeffs(Config(scheme="midpoint", h=0.5)) -> (0.0, 1.0, 1.0, 0.5);
             _coeffs(Config(scheme="leapfrog", h=1.0)) -> (2.0, -1.0, 1.0, 0.5)
    """
    if cfg.scheme == "midpoint":
        return 0.0, 1.0, 2.0 * cfg.h, cfg.h
    if cfg.scheme == "leapfrog":
        return 2.0, -1.0, cfg.h * cfg.h, cfg.h * cfg.h / 2
    raise ValueError(cfg.scheme)


def _combine(cfg, a, b, fb):
    """Next state c from the previous two (a, b) and the block increment fb = f(b), in the stream dtype."""
    al, be, ga, _ = _coeffs(cfg)
    fb = fb.to(b.dtype)
    if cfg.scheme == "midpoint":
        return a + ga * fb
    return 2.0 * b - a + ga * fb                                            # leapfrog (alpha=2, beta=-1)


def _uncombine(cfg, b, c, fb):
    """Exact inverse of _combine: recover a from (b, c) and fb = f(b)."""
    al, be, ga, _ = _coeffs(cfg)
    fb = fb.to(b.dtype)
    if cfg.scheme == "midpoint":
        return c - ga * fb
    return 2.0 * b - c + ga * fb


def run_stack(model, x, keep=None):
    """Run the residual stack forward with whatever autograd mode is active.

    x: (B, T, D) in the stream dtype. Returns (out, final_states) where final_states is the pair the
    reversible backward starts from. keep: optional list that receives every intermediate state
    (detached clones), used only by the reconstruction-error diagnostics.

    Example: m = RevLM(Config(scheme="midpoint")); out, (a, b) = run_stack(m, torch.zeros(1, 4, 384))
             -> out is b = p^L,  a = p^{L-1}
    """
    cfg, blocks = model.cfg, model.blocks
    if cfg.scheme == "euler":
        p = x
        for blk in blocks:
            p = p + blk.delta(p).to(p.dtype)
            if keep is not None:
                keep.append(p.detach().clone())
        return p, (p, p)
    if cfg.scheme == "hamiltonian":
        p, q = x, x
        for blk in blocks:
            q = q + blk.attn_delta(p).to(q.dtype)
            p = p + blk.mlp_delta(q).to(p.dtype)
            if keep is not None:
                keep.append((p.detach().clone(), q.detach().clone()))
        return 0.5 * (p + q), (p, q)
    _, _, _, c0 = _coeffs(cfg)
    if cfg.start == "ghost":
        a, b, k0 = x, x, 0
    else:
        a, b, k0 = x, x + c0 * blocks[0].delta(x).to(x.dtype), 1
    if keep is not None:
        keep.extend([a.detach().clone(), b.detach().clone()])
    for blk in blocks[k0:]:
        a, b = b, _combine(cfg, a, b, blk.delta(b))
        if keep is not None:
            keep.append(b.detach().clone())
    return b, (a, b)


# ------------------------------------------------------------------------ the reversible autograd
def _vjp(out, inputs, grad_out):
    """Vector-Jacobian product: d<out, grad_out>/d inputs, for a small graph built inside backward."""
    return torch.autograd.grad(out, inputs, grad_out.to(out.dtype), allow_unused=True)


def _rel(a, b):
    """Relative error ||a - b|| / ||b|| as a float. Example: _rel(torch.tensor([1., 2.]), torch.tensor([1., 2.])) -> 0.0"""
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


class RevStack(torch.autograd.Function):
    """out = residual stack(x), storing only the last two states; backward rebuilds the rest.

    Inputs: x (B, T, D), the model (for its blocks and config), then every block parameter (so autograd
    knows the parameters are inputs and accepts the gradients returned for them).
    Memory kept between forward and backward: 2 states of N*D numbers, independent of L.
    Compute: each block's increment is evaluated once more in backward; that single evaluation both
    rebuilds the earlier state and provides the graph for the vector-Jacobian product.

    Example: m = RevLM(Config(scheme="midpoint", reversible=True)); x = torch.randn(1, 8, 384, requires_grad=True)
             RevStack.apply(x, m, *m.block_params).sum().backward()   -> x.grad and every block .grad filled
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, model, *params):
        keep = [] if model.debug else None
        out, (a, b) = run_stack(model, x, keep)                 # autograd is off inside Function.forward
        ctx.model, ctx.keep = model, keep
        ctx.x_ref = x.detach().clone() if model.debug else None
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, g_out):
        model = ctx.model
        cfg, blocks = model.cfg, model.blocks
        a, b = ctx.saved_tensors
        index = model.param_index                                # id(param) -> position in *params
        grads = [None] * len(index)

        def add(params, gs):
            for p, g in zip(params, gs):
                if g is not None:
                    i = index[id(p)]
                    grads[i] = g if grads[i] is None else grads[i] + g

        rebuilt = []                                             # diagnostics only (model.debug)
        with torch.enable_grad():
            if cfg.scheme == "hamiltonian":
                p, q = a, b
                gp = gq = 0.5 * g_out
                for blk in reversed(blocks):
                    q_ = q.detach().requires_grad_(True)
                    m = blk.mlp_delta(q_)                       # p' = p + M(q')  ->  p = p' - M(q')
                    p = p - m.detach().to(p.dtype)
                    mp = blk.mlp_params()
                    g = _vjp(m, [q_] + mp, gp)
                    gq = gq + g[0].to(gq.dtype)
                    add(mp, g[1:])
                    p_ = p.detach().requires_grad_(True)
                    at = blk.attn_delta(p_)                     # q' = q + A(p)   ->  q = q' - A(p)
                    q = q - at.detach().to(q.dtype)
                    ap = blk.attn_params()
                    g = _vjp(at, [p_] + ap, gq)
                    gp = gp + g[0].to(gp.dtype)
                    add(ap, g[1:])
                    if ctx.keep is not None:
                        rebuilt.append((p.detach(), q.detach()))
                gx = gp + gq
                if ctx.keep is not None:
                    # rebuilt[j] = state before block L-1-j; its stored copy is keep[L-2-j] (keep[i] = after block i)
                    model.recon_err = [max(_rel(p, ctx.x_ref), _rel(q, ctx.x_ref))] + [
                        max(_rel(pr, pk), _rel(qr, qk)) for (pr, qr), (pk, qk)
                        in zip(reversed(rebuilt[:-1]), ctx.keep[:len(blocks) - 1])]
            elif cfg.scheme in ("midpoint", "leapfrog"):
                al, be, ga, c0 = _coeffs(cfg)
                k0 = 0 if cfg.start == "ghost" else 1
                prev, cur = a, b                                 # (p^{l-1}, p^l), walking down
                gprev, gcur = torch.zeros_like(g_out), g_out
                for blk in reversed(blocks[k0:]):
                    b_ = prev.detach().requires_grad_(True)
                    fb = blk.delta(b_)                           # the one extra forward of this block
                    a_rec = _uncombine(cfg, prev, cur, fb.detach())
                    bp = list(blk.parameters())
                    g = _vjp(fb, [b_] + bp, ga * gcur)
                    add(bp, g[1:])
                    gprev, gcur = be * gcur, gprev + al * gcur + g[0].to(gcur.dtype)
                    prev, cur = a_rec, prev
                    if ctx.keep is not None:
                        rebuilt.append(prev.detach())
                if k0 == 1:                                      # undo p^1 = x + c0 f_0(x) with plain autograd
                    x_ = prev.detach().requires_grad_(True)
                    f0 = blocks[0].delta(x_)
                    bp = list(blocks[0].parameters())
                    g = _vjp(f0, [x_] + bp, c0 * gcur)
                    add(bp, g[1:])
                    gx = gprev + gcur + g[0].to(gcur.dtype)
                else:
                    gx = gprev + gcur
                if ctx.keep is not None:
                    # keep = [p^{-1}, p^0, ..., p^L] (ghost) or [p^0, p^1, ..., p^L] (euler start), bottom-up.
                    # rebuilt, reversed, = every state below the stored top pair, also bottom-up.
                    model.recon_err = [_rel(r, s) for r, s in zip(reversed(rebuilt), ctx.keep[:-2])]
            else:
                raise ValueError("the euler scheme is not reversible; use reversible=False")
        return (gx.to(g_out.dtype), None, *grads)


# ----------------------------------------------------------------------------------------- the LM
def _ce_sum(h, w, t):
    """Summed cross-entropy of one chunk: h (n, D), w (V, D), t (n,) -> scalar. Logits are fp32."""
    return F.cross_entropy(F.linear(h, w).float(), t, reduction="sum")


class RevLM(nn.Module):
    """Decoder-only LM (Llama-style block, tied embedding) with a selectable residual wiring.

    forward(idx (B, T), targets (B, T) or None) -> mean cross-entropy (scalar) or logits (B, T, V).

    Example: m = RevLM(Config()); m(torch.randint(0, 8192, (2, 16)), torch.randint(0, 8192, (2, 16)))
             -> tensor(9.01...) at initialisation  (about ln 8192 = 9.011)
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.scheme in SCHEMES, cfg.scheme
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.debug, self.recon_err = False, None
        self._init()
        self.block_params = [p for blk in self.blocks for p in blk.parameters()]
        self.param_index = {id(p): i for i, p in enumerate(self.block_params)}

    def _init(self):
        """Embedding std 0.02 (tied head -> first logits small, first loss ~ ln V); hidden matrices
        std 1/sqrt(fan_in); the two residual-output matrices (attn.proj, mlp.w2) further / sqrt(2L)."""
        L = self.cfg.n_layer
        nn.init.normal_(self.emb.weight, std=0.02)
        for name, p in self.blocks.named_parameters():
            if p.dim() == 2:
                std = p.shape[1] ** -0.5
                if name.endswith(("proj.weight", "w2.weight")):
                    std /= math.sqrt(2 * L)
                nn.init.normal_(p, std=std)

    def stack(self, x):
        """Residual stack on x (B, T, D) -> (B, T, D), using RevStack when reversible and training."""
        x = x.to(DTYPES[self.cfg.stream])
        if self.cfg.reversible and torch.is_grad_enabled():
            return RevStack.apply(x, self, *self.block_params)
        return run_stack(self, x)[0]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.emb(idx) * self.cfg.emb_mult if self.cfg.emb_mult != 1.0 else self.emb(idx)   # (B, T, D)
        hid = self.norm(self.stack(x).to(self.norm.weight.dtype))                 # (B, T, D)
        w = self.emb.weight                                                       # (V, D), tied head
        if targets is None:
            return F.linear(hid, w).float()                                       # (B, T, V)
        hid, t = hid.reshape(B * T, -1), targets.reshape(-1)                      # (N, D), (N,)
        n, c = hid.shape[0], self.cfg.ce_chunk
        if c <= 0 or not torch.is_grad_enabled():
            if c <= 0:
                return _ce_sum(hid, w, t) / n
            return sum(_ce_sum(hid[i:i + c], w, t[i:i + c]) for i in range(0, n, c)) / n
        # Each chunk's logits (c, V) are rebuilt in backward instead of being stored for all N tokens.
        total = sum(checkpoint(_ce_sum, hid[i:i + c], w, t[i:i + c], use_reentrant=False)
                    for i in range(0, n, c))
        return total / n


def count_params(model):
    """Parameter count by group. Example: count_params(RevLM(Config())) ->
    {'embedding (tied head)': 3145728, 'blocks': 17702400, 'final norm': 384, 'total': 20848512}"""
    blocks = sum(p.numel() for p in model.blocks.parameters())
    emb, norm = model.emb.weight.numel(), model.norm.weight.numel()
    return {"embedding (tied head)": emb, "blocks": blocks, "final norm": norm, "total": emb + blocks + norm}


# ------------------------------------------------------------------------------------ training
def make_optimizer(model, lr):
    """AdamW(0.9, 0.95), weight decay 0.1 on matrices only (embedding included), none on norm gains."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())


def lr_at(step, total, peak, warmup, floor=0.1):
    """Linear warm-up for `warmup` steps, then cosine decay to floor * peak at the last step.

    Example: lr_at(0, 100, 1e-3, 10) -> 1e-4;  lr_at(9, 100, 1e-3, 10) -> 1e-3;  lr_at(99, 100, 1e-3, 10) -> 1e-4
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - 1 - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


@torch.no_grad()
def evaluate(model, val, n_windows, batch=64):
    """Mean next-token loss over the first n_windows rows of val (W, T+1). Plain forward, no stored graph."""
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, n_windows, batch):
        w = val[i:min(i + batch, n_windows)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(w[:, :-1], w[:, 1:])
        tot += loss.item() * w.shape[0]
        cnt += w.shape[0]
    model.train()
    return tot / cnt


def train(cfg, train_tok, val, *, batch, seq=512, total_tokens=50_000_000, lr=1.5e-3, warmup_frac=0.02,
          eval_every=100, eval_windows=256, final_eval_windows=1024, log_every=25, seed=1337, quiet=False,
          keep_model=False):
    """Train one model on the token stream for total_tokens, reading it in order (one epoch, no repeats).

    Step k uses tokens [k*B*T, (k+1)*B*T + 1) of train_tok: inputs = first B*T, targets = shifted by one.
    Speed = tokens per second of pure training-step time (forward + backward + clip + optimizer), measured
    after 5 warm-up steps, evaluation excluded. Peak = torch.cuda.max_memory_allocated over the run.

    Returns a dict, e.g. {'label': 'euler', 'batch': 64, 'steps': 1525, 'final_val': 1.6, 'tok_per_s': 2e5,
                          'peak_gib': 5.5, 'hist': [(step, tokens, loss), ...], 'val_hist': [...], ...}
    """
    dev = "cuda"
    torch.manual_seed(seed)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = RevLM(cfg).to(dev)
    opt = make_optimizer(model, lr)
    tps = batch * seq
    steps = total_tokens // tps
    warmup = max(1, int(round(warmup_frac * steps)))
    hist, val_hist = [], []
    t_train, n_timed, t_start, diverged = 0.0, 0, time.time(), False
    torch.cuda.synchronize()
    for step in range(steps):
        if step == 5:
            torch.cuda.synchronize(); t_train, n_timed = 0.0, 0
        t0 = time.perf_counter()
        buf = train_tok[step * tps: step * tps + tps + 1].long()
        x, y = buf[:-1].view(batch, seq), buf[1:].view(batch, seq)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lv = loss.item()                                         # synchronises the GPU
        t_train += time.perf_counter() - t0
        n_timed += 1 if step >= 5 else 0
        hist.append((step + 1, (step + 1) * tps, lv, float(gnorm)))
        if not math.isfinite(lv):
            diverged = True
            if not quiet:
                print(f"  [{cfg.label()}] loss became {lv} at step {step + 1}: stopping this run")
            break
        last = step == steps - 1
        if (step + 1) % eval_every == 0 and not last:
            val_hist.append((step + 1, (step + 1) * tps, evaluate(model, val, eval_windows)))
        if not quiet and ((step + 1) % log_every == 0 or step == 0 or last):
            vtxt = f"   val {val_hist[-1][2]:.4f}" if val_hist and val_hist[-1][0] == step + 1 else ""
            print(f"  [{cfg.label()}] step {step + 1:5d}/{steps}  tokens {(step + 1) * tps / 1e6:6.2f}M  "
                  f"train loss {lv:.4f}  grad norm {float(gnorm):.3f}  lr {opt.param_groups[0]['lr']:.2e}{vtxt}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    final_val = evaluate(model, val, final_eval_windows) if not diverged else float("nan")
    val_hist.append((len(hist), len(hist) * tps, final_val))
    tail = [h[2] for h in hist[-20:]]
    res = {"label": cfg.label(), "scheme": cfg.scheme, "h": cfg.h, "start": cfg.start, "reversible": cfg.reversible,
           "stream": cfg.stream,
           "batch": batch, "seq": seq, "tokens_per_step": tps, "steps": len(hist), "planned_steps": steps,
           "lr": lr, "warmup_steps": warmup, "final_train": sum(tail) / len(tail), "final_val": final_val,
           "tok_per_s": n_timed * tps / max(t_train, 1e-9), "train_time_s": t_train,
           "wall_s": time.time() - t_start, "peak_gib": peak, "diverged": diverged,
           "hist": hist, "val_hist": val_hist}
    if keep_model:
        res["model"] = model
    else:
        del model, opt
        gc.collect(); torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------- memory probes
def probe(cfg, batch, seq=512, steps=2, timed_steps=0, lr=1e-3):
    """Run `steps` (+ `timed_steps`) training steps on random tokens and report whether they fit.

    Returns {'ok': bool, 'peak_gib': float or None, 'tok_per_s': float or None}. Out-of-memory is caught
    and cleaned up, so a sweep can continue. Random ids are fine: memory does not depend on the values.

    Example: probe(Config(), 16) -> {'ok': True, 'peak_gib': 1.7, 'tok_per_s': None}
    """
    dev = "cuda"
    model = opt = loss = None
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model = RevLM(cfg).to(dev)
        opt = make_optimizer(model, lr)
        x = torch.randint(0, cfg.vocab_size, (batch, seq), device=dev)
        y = torch.randint(0, cfg.vocab_size, (batch, seq), device=dev)
        t_used = 0.0
        for i in range(steps + timed_steps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= steps:
                t_used += time.perf_counter() - t0
        out = {"ok": True, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "tok_per_s": timed_steps * batch * seq / t_used if timed_steps else None}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:   # cuBLAS can report OOM as a RuntimeError
        if "out of memory" not in str(e).lower() and "alloc" not in str(e).lower():
            raise
        out = {"ok": False, "peak_gib": None, "tok_per_s": None}
    del model, opt, loss
    gc.collect(); torch.cuda.empty_cache()
    return out


def max_batch(cfg, seq=512, start=32, step=32, verbose=True):
    """Largest batch (a multiple of `step` sequences) for which 2 training steps fit in GPU memory.

    Doubles from `start` until a probe fails, then bisects between the last fit and the first failure.
    Returns (best_batch, peak_gib_at_best, list of (batch, ok, peak_gib) probes).
    """
    log, lo, lo_peak = [], None, None
    b = start
    while True:
        r = probe(cfg, b, seq)
        log.append((b, r["ok"], r["peak_gib"]))
        if verbose:
            print(f"  [{cfg.label()}] batch {b:5d} x {seq} = {b * seq:>9,} tokens: "
                  + (f"fits, peak {r['peak_gib']:.2f} GiB" if r["ok"] else "out of memory"))
        if not r["ok"]:
            hi = b
            break
        lo, lo_peak, b = b, r["peak_gib"], b * 2
    if lo is None:
        return 0, None, log
    while hi - lo > step:
        mid = (lo + hi) // 2 // step * step
        if mid in (lo, hi):
            break
        r = probe(cfg, mid, seq)
        log.append((mid, r["ok"], r["peak_gib"]))
        if verbose:
            print(f"  [{cfg.label()}] batch {mid:5d} x {seq} = {mid * seq:>9,} tokens: "
                  + (f"fits, peak {r['peak_gib']:.2f} GiB" if r["ok"] else "out of memory"))
        if r["ok"]:
            lo, lo_peak = mid, r["peak_gib"]
        else:
            hi = mid
    return lo, lo_peak, log


# --------------------------------------------------------------------------------- diagnostics
def grad_check(cfg, B=2, T=16, dev="cuda", seed=0, weights=None, batch=None, fp64=False, amp=None):
    """Gradients of the reversible run vs ordinary autograd on the same wiring and the same weights.

    fp64=True: parameters in float64 and no autocast (an exact check of the algorithm; use with stream="fp64").
    fp64=False: fp32 parameters and bf16 autocast, exactly as in training (amp=False: fp32 without autocast).
    weights: optional state_dict (e.g. of a trained model) loaded into both copies; batch: optional
    (idx, targets) pair of (B, T) tensors, otherwise random tokens.
    Returns (max relative gradient error over all parameter tensors, per-state reconstruction errors
    [state 0 = the embedding x, then upward], dict of per-parameter errors).
    In fp64 with dropout 0, an exact scheme should give ~1e-15.

    Example: grad_check(Config(scheme="midpoint", stream="fp64", n_layer=2, d_model=64, n_head=2, d_ff=192), fp64=True)
             -> (2e-16, [1e-16, 0.0, 0.0], {...})
    """
    torch.manual_seed(seed)
    dt = torch.float64 if fp64 else torch.float32                           # parameter dtype
    ref = RevLM(replace(cfg, reversible=False)).to(dev, dt)
    rev = RevLM(replace(cfg, reversible=True)).to(dev, dt)
    if weights is not None:
        ref.load_state_dict(weights)
    rev.load_state_dict(ref.state_dict())
    if batch is None:
        idx = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
        tgt = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    else:
        idx, tgt = batch
    rev.debug = True
    use_amp = (not fp64) if amp is None else amp
    out = []
    for m in (ref, rev):
        torch.manual_seed(seed + 1)                             # same dropout masks in the two forwards
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            m(idx, tgt).backward()
        out.append({n: p.grad.clone() for n, p in m.named_parameters()})
    errs = {n: _rel(out[1][n], out[0][n]) for n in out[0]}
    recon = list(rev.recon_err)                                 # bottom-up: [rebuilt x (or p^-1), next state, ...]
    del ref, rev
    return max(errs.values()), recon, errs


def autograd_repeat_error(cfg, batch, dev="cuda", seed=0, amp=True):
    """Noise floor of a gradient comparison: ordinary autograd run twice on the same weights and tokens.

    Returns the max relative difference over parameter tensors between the two backward passes. Non-zero only
    if some backward kernel is not bit-for-bit deterministic (e.g. atomic adds in an attention backward), and
    then it is the smallest difference any two gradient computations of this model can be expected to show.

    Example: autograd_repeat_error(Config(), (idx, tgt)) -> 0.0 (deterministic) or e.g. 3e-3 (bf16 kernels)
    """
    torch.manual_seed(seed)
    ref = RevLM(replace(cfg, reversible=False)).to(dev)
    idx, tgt = batch
    out = []
    for _ in range(2):
        ref.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            ref(idx, tgt).backward()
        out.append({n: p.grad.clone() for n, p in ref.named_parameters()})
    del ref
    return max(_rel(out[1][n], out[0][n]) for n in out[0])
