"""Ring collectives written by hand on top of point-to-point send/recv, with byte and time counters.

Every collective in Session 12 (all-reduce, reduce-scatter, all-gather) is built here from one primitive:
"send one chunk to my right neighbour while receiving one chunk from my left neighbour". Because we write
every send ourselves, we can count exactly how many bytes each virtual GPU puts on the wire.

Ring layout for W ranks:   0 -> 1 -> 2 -> ... -> W-1 -> 0
(rank r sends to (r + 1) % W and receives from (r - 1) % W)

A flat 1-D tensor of n elements (n divisible by W) is split into W equal chunks of c = n / W elements.

    reduce-scatter : W-1 rounds; afterwards rank r holds chunk r summed over all ranks   sends (W-1) * c elements
    all-gather     : W-1 rounds; afterwards every rank holds every rank's chunk           sends (W-1) * c elements
    all-reduce     : reduce-scatter, then all-gather                                      sends 2 (W-1) * c elements

so an all-reduce of P bytes costs every rank 2 (W-1)/W * P bytes sent, which is the lesson's "about 2P".

Example (W = 4, n = 8, every rank r holds [r, r, r, r, r, r, r, r] in fp32):
    ring_reduce_scatter_(t) on rank 2 -> returns chunk 2 = [0+1+2+3, 0+1+2+3] = [6., 6.]
                                         bytes sent per rank = 3 rounds * 2 elements * 4 bytes = 24
    ring_all_reduce_(t)                 -> every rank holds [6.] * 8;  bytes sent per rank = 48
"""
import time
from collections import defaultdict

import torch
import torch.distributed as dist


class CommStats:
    """What this rank put on the wire, and how long it spent inside collectives.

    bytes_sent[op] : bytes this rank sent for collectives labelled `op` (counted at every single send)
    calls[op]      : number of calls of collectives labelled `op`
    seconds        : wall time spent inside any collective. On real hardware this is transfer time; here it
                     also includes waiting for slower ranks, because a ring round cannot finish before the
                     left neighbour has sent.

    Example: after one ring_all_reduce_ of 1,024 fp32 values on 32 ranks, labelled "all-reduce grads":
        bytes_sent == {"all-reduce grads": 2 * 31 * (1024 / 32) * 4} == {"all-reduce grads": 7936}
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.bytes_sent = defaultdict(int)
        self.calls = defaultdict(int)
        self.seconds = 0.0

    def total_bytes(self):
        return sum(self.bytes_sent.values())

    def save(self):
        """Copy of the counters, so that measurement-only work (evaluation) can be excluded with restore()."""
        return dict(self.bytes_sent), dict(self.calls), self.seconds

    def restore(self, saved):
        b, c, s = saved
        self.bytes_sent, self.calls, self.seconds = defaultdict(int, b), defaultdict(int, c), s

    def as_dict(self):
        return dict(bytes_sent=dict(self.bytes_sent), calls=dict(self.calls), seconds=self.seconds)


STATS = CommStats()


def _wire(t):
    """View bf16/fp16 tensors as int16 on the wire: the bits are unchanged, and every backend can send int16."""
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


def ring_exchange(send_t, recv_t, op):
    """One ring round: send `send_t` to the right neighbour while receiving into `recv_t` from the left one.

    Both calls are non-blocking (isend / irecv), so all W ranks can send at the same moment without deadlock.
    """
    W, r = dist.get_world_size(), dist.get_rank()
    reqs = [dist.isend(_wire(send_t), (r + 1) % W), dist.irecv(_wire(recv_t), (r - 1) % W)]
    for q in reqs:
        q.wait()
    STATS.bytes_sent[op] += send_t.numel() * send_t.element_size()


def _reduce_scatter_rounds(chunks, op):
    """W-1 rounds. In round s, rank r sends chunk (r-s-1) and adds what arrives into chunk (r-s-2).

    The chunk a rank sends in round s is the one it received (and added to) in round s-1, so every send carries
    one more contribution. After round W-2 the chunk received last is (r - W) % W = r: complete on rank r.
    """
    W, r = len(chunks), dist.get_rank()
    buf = torch.empty_like(chunks[0])
    for s in range(W - 1):
        ring_exchange(chunks[(r - s - 1) % W], buf, op)
        chunks[(r - s - 2) % W].add_(buf)


def _all_gather_rounds(chunks, op):
    """W-1 rounds. In round s, rank r forwards chunk (r-s) and receives chunk (r-s-1) straight into place."""
    W, r = len(chunks), dist.get_rank()
    for s in range(W - 1):
        ring_exchange(chunks[(r - s) % W], chunks[(r - s - 1) % W], op)


def _check_flat(flat, W):
    assert flat.dim() == 1 and flat.is_contiguous(), "ring collectives work on flat contiguous tensors"
    assert flat.numel() % W == 0, f"{flat.numel()} elements cannot be split into {W} equal chunks; pad first"


def ring_reduce_scatter_(flat, op="reduce-scatter"):
    """In place. Returns this rank's chunk (a view of `flat`), summed over all ranks.

    Example (W = 2): rank 0 holds [1, 2], rank 1 holds [10, 20]
        -> rank 0 returns [11.], rank 1 returns [22.]       (the other chunk of `flat` is left partly summed)
    """
    W, r = dist.get_world_size(), dist.get_rank()
    STATS.calls[op] += 1
    if W == 1:
        return flat
    _check_flat(flat, W)
    t0 = time.perf_counter()
    chunks = flat.chunk(W)
    _reduce_scatter_rounds(chunks, op)
    STATS.seconds += time.perf_counter() - t0
    return chunks[r]


def ring_all_gather_(flat, op="all-gather"):
    """In place. On entry chunk r of `flat` must hold rank r's data; on exit every chunk is filled on every rank.

    Example (W = 2): rank 0 holds [1, ?], rank 1 holds [?, 22]  ->  both hold [1., 22.]
    """
    W = dist.get_world_size()
    STATS.calls[op] += 1
    if W == 1:
        return flat
    _check_flat(flat, W)
    t0 = time.perf_counter()
    _all_gather_rounds(flat.chunk(W), op)
    STATS.seconds += time.perf_counter() - t0
    return flat


def ring_all_reduce_(t, op="all-reduce"):
    """In place sum over all ranks = reduce-scatter followed by all-gather. Any size (padded internally).

    Example (W = 32): every rank r holds [r]  ->  every rank holds [0 + 1 + ... + 31] = [496.]
    """
    W = dist.get_world_size()
    STATS.calls[op] += 1
    if W == 1:
        return t
    t0 = time.perf_counter()
    assert t.is_contiguous(), "ring_all_reduce_ needs a contiguous tensor"
    flat = t.view(-1)
    n, pad = flat.numel(), (-flat.numel()) % W
    work = torch.cat([flat, flat.new_zeros(pad)]) if pad else flat     # padding: zeros do not change a sum
    chunks = work.chunk(W)
    _reduce_scatter_rounds(chunks, op)
    _all_gather_rounds(chunks, op)
    if pad:
        flat.copy_(work[:n])
    STATS.seconds += time.perf_counter() - t0
    return t
