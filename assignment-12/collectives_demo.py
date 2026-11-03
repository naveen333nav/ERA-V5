"""The virtual GPUs say hello, run the three collectives, and time a ring all-reduce.

    torchrun --standalone --nproc_per_node=32 collectives_demo.py --mode hello       --out hello.json
    torchrun --standalone --nproc_per_node=32 collectives_demo.py --mode collectives --out coll.json
    torchrun --standalone --nproc_per_node=32 collectives_demo.py --mode bench       --out bench.json
"""
import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist

from dist_lib import STATS, ring_all_gather_, ring_all_reduce_, ring_reduce_scatter_


def max_over_ranks(x):
    t = torch.tensor([float(x)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


def hello(W, r):
    """Every rank reports its identity; rank 0 collects the table."""
    info = dict(rank=r, local_rank=int(os.environ.get("LOCAL_RANK", r)), world_size=W, pid=os.getpid(),
                host=socket.gethostname(), threads=torch.get_num_threads(), backend=dist.get_backend())
    table = [None] * W
    dist.all_gather_object(table, info)
    return dict(ranks=table)


def collectives(W, r):
    """The three collectives on numbers that can be checked by hand.

    Rank r holds the vector x_r with W entries, x_r[c] = 100 r + c (so rank 2 of 32 holds 200, 201, ..., 231).
        all-reduce     : every rank gets sum_r x_r[c] = 100 * W(W-1)/2 + W c            (W = 32: 49600 + 32 c)
        reduce-scatter : rank c gets only entry c of that sum
        all-gather     : of each rank's own entry r  ->  every rank gets the full sum again
    """
    out = {}
    x = torch.tensor([100.0 * r + c for c in range(W)])                       # (W,)
    expected = torch.tensor([100.0 * W * (W - 1) / 2 + W * c for c in range(W)])

    # 1. Library all-reduce (gloo's own implementation)
    a = x.clone()
    dist.all_reduce(a)
    out["lib_allreduce_rank0_first4"] = a[:4].tolist() if r == 0 else None
    out["lib_allreduce_maxerr"] = max_over_ranks((a - expected).abs().max())

    # 2. Hand-written ring reduce-scatter: rank r ends with entry r of the sum
    b = x.clone()
    mine = ring_reduce_scatter_(b, "reduce-scatter")
    got = [None] * W
    dist.all_gather_object(got, mine.tolist())
    out["ring_rs_each_rank_gets"] = [g[0] for g in got]
    out["ring_rs_maxerr"] = max(abs(g[0] - float(expected[i])) for i, g in enumerate(got))

    # 3. Hand-written ring all-gather of those slices: every rank holds the full sum = all-reduce result
    ring_all_gather_(b, "all-gather")
    out["rs_then_ag_vs_allreduce_maxerr"] = max_over_ranks((b - a).abs().max())

    # 4. Hand-written ring all-reduce in one call
    c = x.clone()
    ring_all_reduce_(c, "all-reduce")
    out["ring_allreduce_vs_lib_maxerr"] = max_over_ranks((c - a).abs().max())

    # 5. Does this gloo build have a library reduce-scatter? (older gloo builds do not)
    try:
        o = torch.empty(1)
        dist.reduce_scatter_tensor(o, x.clone())
        out["lib_reduce_scatter"] = f"available, max error {max_over_ranks((o - expected[r]).abs().max()):.1f}"
    except Exception as e:
        out["lib_reduce_scatter"] = f"not available in this gloo build ({type(e).__name__})"

    # 6. Bytes on the wire for a bigger message: n = W * 1024 fp32 values = 128 KiB per rank at W = 32
    STATS.reset()
    n = W * 1024
    t = torch.randn(n)
    ring_reduce_scatter_(t.clone(), "reduce-scatter")
    ring_all_gather_(t.clone(), "all-gather")
    ring_all_reduce_(t.clone(), "all-reduce")
    sent = [None] * W
    dist.all_gather_object(sent, dict(STATS.bytes_sent))
    out["bytes_message"] = n * 4
    out["bytes_sent_rank0"] = sent[0]
    out["bytes_sent_all_ranks_equal"] = all(s == sent[0] for s in sent)

    # 7. Averaging gradients = all-reduce then divide by W: check the identity on random "gradients"
    g = torch.randn(1000, generator=torch.Generator().manual_seed(r))
    gs = [None] * W
    dist.all_gather_object(gs, g.tolist())
    true_mean = torch.tensor(gs).mean(0)
    avg = g.clone()
    ring_all_reduce_(avg, "all-reduce")
    avg /= W
    out["average_vs_true_mean_maxerr"] = max_over_ranks((avg - true_mean).abs().max())
    return out


def bench(W, r, reps_small=5, reps_large=3, max_pow=17):
    """Median time of one all-reduce of W * 2^k fp32 values, hand ring vs gloo's own, k = 0..max_pow."""
    rows = []
    for k in range(max_pow + 1):
        n = W * 2 ** k
        t = torch.randn(n)
        reps = reps_small if k < 12 else reps_large
        res = {}
        for name, fn in (("ring", lambda z: ring_all_reduce_(z, "bench")), ("gloo", dist.all_reduce)):
            times = []
            for _ in range(reps + 1):                                         # first repetition is a warm-up
                z = t.clone()
                dist.barrier()
                t0 = time.perf_counter()
                fn(z)
                times.append(max_over_ranks(time.perf_counter() - t0))         # the slowest rank sets the time
            res[name] = sorted(times[1:])[len(times[1:]) // 2]
        rows.append(dict(n=n, bytes=n * 4, ring_s=res["ring"], gloo_s=res["gloo"]))
        if r == 0:
            print(f"  all-reduce of {n * 4 / 1024:10.1f} KiB : hand ring {res['ring'] * 1e3:9.2f} ms   "
                  f"gloo {res['gloo'] * 1e3:9.2f} ms", flush=True)
    return dict(rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hello", "collectives", "bench"], required=True)
    ap.add_argument("--max_pow", type=int, default=17)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dist.init_process_group("gloo")
    torch.set_num_threads(1)
    W, r = dist.get_world_size(), dist.get_rank()
    out = dict(W=W)
    if a.mode == "hello":
        out.update(hello(W, r))
    elif a.mode == "collectives":
        out.update(collectives(W, r))
    else:
        out.update(bench(W, r, max_pow=a.max_pow))
    if r == 0:
        with open(a.out, "w") as f:
            json.dump(out, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
