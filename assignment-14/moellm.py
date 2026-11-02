"""Dense and Mixture-of-Experts Llama-style LMs, dense -> MoE upcycling, training loop and routing diagnostics.

Architecture (pre-norm Llama block, the same family as Assignments 10-13):

    token ids (B, T)
      -> Embedding (tied with the LM head)                                  x (B, T, D), fp32 residual stream
      -> L blocks:   p = p + Attn(RMSNorm(p))
                     p = p + FFN(RMSNorm(p))
                         FFN = SwiGLU of width F                             (dense model)
                             = MoE: router + E SwiGLU experts of width Fe     (MoE model, k experts per token)
      -> RMSNorm -> logits = h @ Emb^T  (B, T, V) -> cross-entropy

MoE layer, for the N = B*T tokens of a batch (lesson sections 3, 7, 11, 13):

    logits  = x @ Wr^T                          (N, E)   computed in fp32 (router stability, lesson section 7)
    scores  = sigmoid(logits)                   (N, E)   each in (0, 1), no competition (DeepSeek-V3 style;
                                                         score_fn="softmax" gives the Qwen3 / Switch style)
    chosen  = top-k of (scores + bias)          (N, k)   the bias steers only the choice (aux-loss-free balancing)
    weights = scores[chosen] / sum * scale      (N, k)   scale = routed scaling factor (= k here, see upcycle())
    y       = sum_j weights_j * Expert_{chosen_j}(x)

Dispatch is sort-based and dropless: the A = N*k (token, expert) assignments are sorted by expert id, each expert
runs once on its contiguous block of rows (any length, so no token is dropped), and the weighted outputs are
scatter-added back to their tokens. capacity_factor > 0 switches on token dropping (used for analysis only).

Notation in docstrings and shape comments:
    B sequences per batch   T tokens per sequence   D d_model    H heads      hd = D / H
    F dense FFN width       E experts               Fe expert width          k experts per token
    V vocabulary size       L layers                N = B * T tokens         A = N * k assignments
"""
import gc
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 10              # L
    n_head: int = 6                # H   (hd = 384 / 6 = 64)
    d_model: int = 384             # D
    d_ff: int = 1024               # F, width of the dense SwiGLU
    max_seq: int = 512             # T the RoPE tables are built for
    rope_base: float = 10000.0
    # ---- MoE (n_expert = 0 means the dense model)
    n_expert: int = 0              # E
    expert_ff: int = 256           # Fe
    top_k: int = 4                 # k
    route_scale: float = 4.0       # routed scaling factor: the k weights of a token add up to this
    score_fn: str = "sigmoid"      # "sigmoid" (DeepSeek-V3, Kimi K2, GLM-4.5) or "softmax" (Qwen3, Switch)
    router_std_mult: float = 0.1   # router init std = mult / sqrt(D)  (Switch Transformer: 0.1 x the usual scale)
    bias_gamma: float = 1e-3       # aux-loss-free bias update speed per step; 0 = no bias balancing
    aux_alpha: float = 0.0         # Switch batch-level auxiliary loss coefficient
    seq_aux_alpha: float = 1e-4    # sequence-level auxiliary loss coefficient (DeepSeek-V3 style safety net)
    capacity_factor: float = 0.0   # 0 = dropless; > 0 drops assignments beyond capacity (analysis only)

    @property
    def is_moe(self):
        return self.n_expert > 0

    def label(self):
        """Short run name. Example: Config().label() -> 'dense F=1024';
        Config(n_expert=32).label() -> 'MoE 32x256 top-4, bias 0.001 + seq-aux 0.0001'"""
        if not self.is_moe:
            return f"dense F={self.d_ff}"
        bal = []
        if self.bias_gamma > 0:
            bal.append(f"bias {self.bias_gamma:g}")
        if self.aux_alpha > 0:
            bal.append(f"aux {self.aux_alpha:g}")
        if self.seq_aux_alpha > 0:
            bal.append(f"seq-aux {self.seq_aux_alpha:g}")
        return f"MoE {self.n_expert}x{self.expert_ff} top-{self.top_k}, " + (" + ".join(bal) or "no balancing")


# ----------------------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * w.

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
    """Rotate each (even, odd) feature pair of x (B, H, T, hd) by its position's angle. Norm-preserving.

    Example: one head, hd=2, position 1, angle 1 rad: (1, 0) -> (cos 1, sin 1) = (0.5403, 0.8415)
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]                                       # (B, H, T, hd/2) each
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)     # (B, H, T, hd/2, 2)
    return out.flatten(-2)                                                    # (B, H, T, hd)


class Attention(nn.Module):
    """Causal multi-head self-attention with RoPE, no biases. (B, T, D) -> (B, T, D).

    Example: Attention(Config())(torch.randn(2, 16, 384)).shape -> torch.Size([2, 16, 384])
    """

    def __init__(self, cfg):
        super().__init__()
        self.H, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        cos, sin = rope_tables(self.hd, cfg.max_seq, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)   # 3 x (B, H, T, hd)
        cos, sin = self.cos[:T].to(q.dtype), self.sin[:T].to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)                     # (B, H, T, hd)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))                            # (B, T, D)


class SwiGLU(nn.Module):
    """Dense gated feed-forward: w2( silu(w1 x) * w3 x ), hidden width F. (B, T, D) -> (B, T, D).

    It is a sum over its F hidden neurons, which is what makes partition upcycling exact:
        FFN(x) = sum_j  w2[:, j] * silu(w1[j] . x) * (w3[j] . x)

    Example: SwiGLU(Config())(torch.randn(2, 16, 384)).shape -> torch.Size([2, 16, 384])
    """

    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)    # gate  (F, D)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)    # up    (F, D)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)    # down  (D, F)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ------------------------------------------------------------------------------------------- MoE
def switch_aux(f, P, alpha):
    """Switch Transformer load-balancing loss  alpha * E * sum_i f_i P_i  over the last axis.

    f: share of assignments sent to each expert (no gradient), P: mean router probability per expert.
    Example (E = 4): even  f = P = [.25]*4      -> alpha * 4 * 0.25 = alpha        (0.01 for alpha = 0.01)
                     collapsed f = P = [1,0,0,0] -> alpha * 4 * 1    = 4 * alpha   (0.04)
    """
    E = f.shape[-1]
    return alpha * E * (f * P).sum(-1)


def dispatch_plan(flat_e, E):
    """Group the A assignments by expert (sort-based dispatch).

    flat_e (A,): expert id of assignment a = token * k + j.
    Returns order (A,): assignment indices sorted by expert, token order kept inside each expert (stable sort);
            counts (E,): assignments per expert;  pos (A,): arrival rank of each sorted assignment in its expert.
    Example: flat_e = [2, 0, 2, 1, 0, 2], E = 3 -> order = [1, 4, 3, 0, 2, 5], counts = [2, 1, 3],
             pos = [0, 1, 0, 0, 1, 2]   (with capacity 2, the assignment with pos 2, token 5, would be dropped)
    """
    order = torch.argsort(flat_e, stable=True)                                # (A,)
    counts = torch.bincount(flat_e, minlength=E)                              # (E,)
    starts = torch.cumsum(counts, 0) - counts                                 # (E,) first sorted row of each expert
    pos = torch.arange(flat_e.numel(), device=flat_e.device) - starts[flat_e[order]]   # (A,)
    return order, counts, pos


class MoE(nn.Module):
    """Router + E SwiGLU experts of width Fe, top-k, dropless sort-based dispatch. (B, T, D) -> (B, T, D).

    Expert e:  E_e(x) = w2[e] ( silu(w1[e] x) * w3[e] x ),   w1[e], w3[e]: (Fe, D),  w2[e]: (D, Fe).
    After every forward pass the layer keeps:  last_counts (E,) assignments per expert in that batch,
    aux (scalar or None) its auxiliary loss, last_topi (N, k) the chosen experts if record is True,
    last_dropped the share of assignments dropped by a capacity limit (0 when dropless).

    Example: MoE(Config(n_expert=32))(torch.randn(2, 16, 384)).shape -> torch.Size([2, 16, 384]),
             and .last_counts.sum() -> 2 * 16 * 4 = 128
    """

    def __init__(self, cfg):
        super().__init__()
        E, Fe, D = cfg.n_expert, cfg.expert_ff, cfg.d_model
        self.cfg, self.E, self.k = cfg, E, cfg.top_k
        self.router = nn.Parameter(torch.empty(E, D))                         # Wr (E, D), fp32
        self.w1 = nn.Parameter(torch.empty(E, Fe, D))                         # gate
        self.w3 = nn.Parameter(torch.empty(E, Fe, D))                         # up
        self.w2 = nn.Parameter(torch.empty(E, D, Fe))                         # down
        self.register_buffer("bias", torch.zeros(E))                          # selection bias, not trained by gradient
        self.aux, self.last_counts, self.last_topi, self.last_dropped, self.record = None, None, None, 0.0, False

    def route(self, xf):
        """xf (N, D) -> probs (N, E), chosen experts topi (N, k), their weights topw (N, k), all fp32.

        probs = scores normalised to sum to 1 over the experts (the P_i of the auxiliary losses; for softmax this
        is the softmax itself). Example, sigmoid, 8 experts, k = 2, logits [2.0, .5, 1.2, -.3, .9, 1.8, -1., .1]:
        scores [.881, .622, .769, .426, .711, .858, .269, .525] -> topi [0, 5], topw [.507, .493] * route_scale
        """
        with torch.autocast(device_type=xf.device.type, enabled=False):
            logits = F.linear(xf.float(), self.router.float())                # (N, E) in fp32
        if self.cfg.score_fn == "softmax":
            scores = logits.softmax(-1)                                       # (N, E), sum to 1
        else:
            scores = torch.sigmoid(logits)                                    # (N, E), each in (0, 1)
        topi = (scores + self.bias).topk(self.k, dim=-1).indices              # (N, k) choice uses the bias
        topw = scores.gather(-1, topi)                                        # (N, k) weights do not
        topw = topw / topw.sum(-1, keepdim=True) * self.cfg.route_scale
        probs = scores / scores.sum(-1, keepdim=True)                         # (N, E)
        return probs, topi, topw

    def forward(self, x):
        B, T, D = x.shape
        N, k, E = B * T, self.k, self.E
        xf = x.reshape(N, D)                                                  # (N, D)
        probs, topi, topw = self.route(xf)
        flat_e = topi.reshape(-1)                                             # (A,)
        order, counts, pos = dispatch_plan(flat_e, E)
        tok = order // k                                                      # (A,) token of each sorted assignment
        w = topw.reshape(-1)[order]                                           # (A,) its router weight
        if self.cfg.capacity_factor > 0:                                      # token dropping, first come first served
            cap = math.ceil(N * k / E * self.cfg.capacity_factor)
            keep = pos < cap
            w = w * keep
            self.last_dropped = 1.0 - keep.float().mean().item()
        else:
            self.last_dropped = 0.0
        xs = xf[tok]                                                          # (A, D) rows grouped by expert
        w1s, w3s, w2s = self.w1.unbind(0), self.w3.unbind(0), self.w2.unbind(0)   # unbind: one cheap backward op
        outs, s = [], 0
        for e, c in enumerate(counts.tolist()):                               # one matmul chain per expert
            if c:
                h = xs[s:s + c]                                               # (c, D)
                outs.append(F.linear(F.silu(F.linear(h, w1s[e])) * F.linear(h, w3s[e]), w2s[e]))   # (c, D)
                s += c
        out = torch.cat(outs)                                                 # (A, D)
        y = torch.zeros(N, D, device=x.device, dtype=torch.float32)
        y.index_add_(0, tok, out.float() * w[:, None])                        # combine: k weighted outputs per token
        self.last_counts = counts.detach()
        if self.record:
            self.last_topi = topi.detach()
        cfg = self.cfg
        if self.training and (cfg.aux_alpha > 0 or cfg.seq_aux_alpha > 0):
            aux = probs.new_zeros(())
            if cfg.aux_alpha > 0:                                             # over the whole batch
                aux = aux + switch_aux(counts.float() / (N * k), probs.mean(0), cfg.aux_alpha)
            if cfg.seq_aux_alpha > 0:                                         # per sequence, then averaged
                ones = torch.ones(B, T * k, device=x.device)
                f_seq = torch.zeros(B, E, device=x.device).scatter_add_(1, topi.view(B, T * k), ones) / (T * k)
                aux = aux + switch_aux(f_seq, probs.view(B, T, E).mean(1), cfg.seq_aux_alpha).mean()
            self.aux = aux
        else:
            self.aux = None
        return y.to(x.dtype).view(B, T, D)

    @torch.no_grad()
    def update_bias(self):
        """Aux-loss-free balancing step: b_i += gamma * sign(mean load - load_i), from the last batch's counts.

        Example: counts [5, 1, 1, 1], gamma 0.001 -> bias change [-0.001, +0.001, +0.001, +0.001]
        """
        if self.cfg.bias_gamma > 0 and self.last_counts is not None:
            c = self.last_counts.float()
            self.bias += self.cfg.bias_gamma * torch.sign(c.mean() - c)


class Block(nn.Module):
    """Pre-norm block: p + Attn(LN1 p), then + FFN(LN2 p), with FFN = SwiGLU (dense) or MoE."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = MoE(cfg) if cfg.is_moe else SwiGLU(cfg)

    def forward(self, p):                                                     # (B, T, D) -> (B, T, D)
        p = p + self.attn(self.ln1(p))
        return p + self.ffn(self.ln2(p))


class LM(nn.Module):
    """Decoder-only LM with tied embedding; dense or MoE feed-forward blocks.

    forward(idx (B, T), targets (B, T) or None) -> mean cross-entropy (scalar) or logits (B, T, V) in fp32.
    After a training forward, aux_loss() returns the sum of the MoE layers' auxiliary losses (or None).

    Example: m = LM(Config()); m(torch.randint(0, 8192, (2, 16)), torch.randint(0, 8192, (2, 16)))
             -> tensor(9.01...) at initialisation  (about ln 8192 = 9.011)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self._init()

    def _init(self):
        """Embedding std 0.02; input matrices std 1/sqrt(fan_in); residual-output matrices (attn.proj, w2) further
        / sqrt(2L). MoE: w2 uses fan_in k*Fe (the k chosen experts together play the role of one F-wide FFN),
        router std router_std_mult / sqrt(D), so the first router scores are almost equal (sigmoid ~ 0.5)."""
        c, L = self.cfg, self.cfg.n_layer
        nn.init.normal_(self.emb.weight, std=0.02)
        for b in self.blocks:
            nn.init.normal_(b.attn.qkv.weight, std=c.d_model ** -0.5)
            nn.init.normal_(b.attn.proj.weight, std=c.d_model ** -0.5 / math.sqrt(2 * L))
            f = b.ffn
            if isinstance(f, MoE):
                nn.init.normal_(f.w1, std=c.d_model ** -0.5)
                nn.init.normal_(f.w3, std=c.d_model ** -0.5)
                nn.init.normal_(f.w2, std=(c.top_k * c.expert_ff) ** -0.5 / math.sqrt(2 * L))
                nn.init.normal_(f.router, std=c.router_std_mult * c.d_model ** -0.5)
            else:
                nn.init.normal_(f.w1.weight, std=c.d_model ** -0.5)
                nn.init.normal_(f.w3.weight, std=c.d_model ** -0.5)
                nn.init.normal_(f.w2.weight, std=c.d_ff ** -0.5 / math.sqrt(2 * L))

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def aux_loss(self):
        terms = [m.aux for m in self.moe_layers() if m.aux is not None]
        return sum(terms) if terms else None

    def forward(self, idx, targets=None):
        x = self.emb(idx)                                                     # (B, T, D)
        for b in self.blocks:
            x = b(x)
        logits = F.linear(self.norm(x), self.emb.weight)                      # (B, T, V), tied head
        if targets is None:
            return logits.float()
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))


def count_params(model):
    """Parameter count by group, with the active count (what one token uses).

    Example: count_params(LM(Config())) -> {'embedding (tied head)': 3145728, 'attention': 5898240,
             'norms': 8064, 'FFN / experts': 11796480, 'router': 0, 'total': 20848512, 'active': 20848512}
    """
    c = model.cfg
    emb = model.emb.weight.numel()
    attn = sum(p.numel() for b in model.blocks for p in b.attn.parameters())
    norms = sum(p.numel() for n, p in model.named_parameters() if ".ln" in n or n == "norm.weight")
    if c.is_moe:
        ffn = sum(m.w1.numel() + m.w3.numel() + m.w2.numel() for m in model.moe_layers())
        router = sum(m.router.numel() for m in model.moe_layers())
        active_ffn = ffn * c.top_k // c.n_expert
    else:
        ffn = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
        router, active_ffn = 0, ffn
    total = emb + attn + norms + ffn + router
    return {"embedding (tied head)": emb, "attention": attn, "norms": norms, "FFN / experts": ffn,
            "router": router, "total": total, "active": total - ffn + active_ffn}


# ------------------------------------------------------------------------------------- upcycling
@torch.no_grad()
def upcycle(dense, moe_cfg, method="partition", r=0.0, seed=0):
    """Build an MoE LM from a trained dense LM (lesson section 15). Everything outside the FFNs (embedding,
    attention, norms) is copied; each router gets a fresh, small random init and a zero bias.

    method "copy":      every expert is a full copy of the dense FFN (needs expert_ff == d_ff).
    method "partition": the F neurons are cut into S = F / Fe slices; expert e takes slice e % S (its "family"),
                        so experts e, e+S, e+2S, ... start as clones of the same slice.
    r > 0 ("drop-upcycling"): in every expert a random fraction r of its Fe neurons is redrawn (its w1, w3 rows
    and w2 column) from N(mean, std) of the corresponding dense matrix; each expert redraws a different subset.

    Returns (moe_model, family), family (E,) = the dense slice each expert came from.
    Example: dense F = 1024, moe_cfg E = 32, Fe = 256, "partition" -> family = [0, 1, 2, 3, 0, 1, 2, 3, ...]
             (4 families of 8 clones);  "copy" with E = 8, Fe = 1024 -> family = [0] * 8
    """
    dc, E, Fe = dense.cfg, moe_cfg.n_expert, moe_cfg.expert_ff
    assert (dc.d_model, dc.n_layer, dc.n_head, dc.vocab_size) == \
           (moe_cfg.d_model, moe_cfg.n_layer, moe_cfg.n_head, moe_cfg.vocab_size), "shapes must match"
    if method == "copy":
        assert Fe == dc.d_ff, "copy needs expert_ff == d_ff"
        S = 1
    else:
        assert method == "partition" and dc.d_ff % Fe == 0, (method, dc.d_ff, Fe)
        S = dc.d_ff // Fe
    dev = next(dense.parameters()).device
    torch.manual_seed(seed)                                                   # router init
    moe = LM(moe_cfg).to(dev)
    sd = {n: v for n, v in dense.state_dict().items() if ".ffn." not in n}
    missing, unexpected = moe.load_state_dict(sd, strict=False)
    assert not unexpected and all(".ffn." in n for n in missing), (missing, unexpected)
    family = torch.arange(E) % S
    g = torch.Generator().manual_seed(seed)
    n_new = int(round(r * Fe))
    for bd, bm in zip(dense.blocks, moe.blocks):
        W1, W3, W2 = bd.ffn.w1.weight, bd.ffn.w3.weight, bd.ffn.w2.weight    # (F, D), (F, D), (D, F)
        m = bm.ffn
        for e in range(E):
            sl = slice(int(family[e]) * Fe, (int(family[e]) + 1) * Fe)
            m.w1[e].copy_(W1[sl]); m.w3[e].copy_(W3[sl]); m.w2[e].copy_(W2[:, sl])
            if n_new:
                idx = torch.randperm(Fe, generator=g)[:n_new].to(dev)        # neurons of this expert to redraw
                D = W1.shape[1]
                m.w1[e][idx] = (torch.randn(n_new, D, generator=g).to(dev) * W1.std() + W1.mean())
                m.w3[e][idx] = (torch.randn(n_new, D, generator=g).to(dev) * W3.std() + W3.mean())
                m.w2[e][:, idx] = (torch.randn(D, n_new, generator=g).to(dev) * W2.std() + W2.mean())
    return moe, family


# ------------------------------------------------------------------------------------ training
def make_optimizer(model, lr):
    """AdamW(0.9, 0.95), weight decay 0.1 on matrices (embedding, experts, router included), none on norm gains."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())


def lr_at(step, total, peak, warmup, decay_frac=0.2, floor=0.1):
    """Warmup-stable-decay (WSD): linear warm-up over `warmup` steps, constant peak, then a linear decay to
    floor * peak over the last decay_frac of the steps. decay_frac = 0: no decay (a "mid-training" checkpoint).

    Example: lr_at(0, 100, 1e-3, 10) -> 1e-4;  lr_at(50, 100, 1e-3, 10) -> 1e-3;  lr_at(99, 100, 1e-3, 10) -> 1e-4;
             lr_at(99, 100, 1e-3, 10, decay_frac=0) -> 1e-3
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    d0 = total - int(round(decay_frac * total))
    if decay_frac <= 0 or step < d0:
        return peak
    return peak * (1 - (1 - floor) * (step - d0 + 1) / (total - d0))


@torch.no_grad()
def evaluate(model, val, n_windows, batch=64):
    """Mean next-token loss over the first n_windows rows of val (W, T+1), bf16 autocast, no auxiliary loss."""
    was = model.training
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, n_windows, batch):
        w = val[i:min(i + batch, n_windows)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(w[:, :-1], w[:, 1:])
        tot += loss.item() * w.shape[0]
        cnt += w.shape[0]
    model.train(was)
    return tot / cnt


def train(model, train_tok, val, *, start_token=0, n_tokens, batch=64, seq=512, lr=1.5e-3, warmup_frac=0.02,
          decay_frac=0.2, eval_every=50, eval_windows=1024, final_eval_windows=1024, log_every=50, seed=1337,
          label=None, quiet=False):
    """Train `model` in place on train_tok[start_token : start_token + n_tokens], read in order, once.

    Step s uses tokens [start + s*B*T, start + (s+1)*B*T + 1): inputs = first B*T, targets shifted by one.
    Fresh AdamW, WSD schedule (see lr_at). Loss = LM loss + MoE auxiliary losses; only the LM loss is logged
    as "train loss". After each optimizer step every MoE layer updates its selection bias.
    Speed = tokens per second of pure training-step time (eval excluded), measured after 5 warm-up steps.

    Returns a dict, e.g. {'label': ..., 'steps': 610, 'val_start': 2.4, 'final_val': 1.9, 'tok_per_s': 5e4,
        'peak_gib': 9.1, 'hist': [(step, global_tokens, lm_loss, aux_loss, grad_norm), ...],
        'val_hist': [(step, global_tokens, val_loss), ...], 'loads': array (steps, L, E) or None}
    """
    dev = "cuda"
    label = label or model.cfg.label()
    torch.manual_seed(seed)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = make_optimizer(model, lr)
    moes = model.moe_layers()
    tps = batch * seq
    steps = n_tokens // tps
    warmup = max(1, int(round(warmup_frac * steps)))
    v0 = evaluate(model, val, eval_windows)
    hist, val_hist, loads = [], [(0, start_token, v0)], []
    t_train, n_timed, t_start, diverged = 0.0, 0, time.time(), False
    if not quiet:
        print(f"  [{label}] {steps} steps of {tps:,} tokens, tokens {start_token / 1e6:.0f}M -> "
              f"{(start_token + steps * tps) / 1e6:.0f}M;  val loss before the first step {v0:.4f}")
    for step in range(steps):
        if step == 5:
            torch.cuda.synchronize(); t_train, n_timed = 0.0, 0
        t0 = time.perf_counter()
        a = start_token + step * tps
        buf = train_tok[a: a + tps + 1].long()
        x, y = buf[:-1].view(batch, seq), buf[1:].view(batch, seq)            # (B, T) each
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, steps, lr, warmup, decay_frac)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        aux = model.aux_loss()
        (loss + aux if aux is not None else loss).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if moes:
            loads.append(torch.stack([m.last_counts for m in moes]).cpu().numpy().astype(np.int32))   # (L, E)
            for m in moes:
                m.update_bias()
        lv = loss.item()                                                      # synchronises the GPU
        t_train += time.perf_counter() - t0
        n_timed += 1 if step >= 5 else 0
        g_tok = a + tps
        hist.append((step + 1, g_tok, lv, aux.item() if aux is not None else 0.0, float(gnorm)))
        if not math.isfinite(lv):
            diverged = True
            print(f"  [{label}] loss became {lv} at step {step + 1}: stopping this run")
            break
        last = step == steps - 1
        if (step + 1) % eval_every == 0 and not last:
            val_hist.append((step + 1, g_tok, evaluate(model, val, eval_windows)))
        if not quiet and ((step + 1) % log_every == 0 or step == 0 or last):
            vtxt = f"   val {val_hist[-1][2]:.4f}" if val_hist[-1][0] == step + 1 else ""
            mv = ""
            if moes:
                c = loads[-1].astype(float)
                mv = f"  MaxVio (mean over layers) {((c.max(1) - c.mean(1)) / c.mean(1)).mean():.2f}"
            print(f"  [{label}] step {step + 1:5d}/{steps}  tokens {g_tok / 1e6:6.2f}M  train loss {lv:.4f}  "
                  f"grad norm {float(gnorm):.3f}  lr {opt.param_groups[0]['lr']:.2e}{mv}{vtxt}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    final_val = evaluate(model, val, final_eval_windows) if not diverged else float("nan")
    val_hist.append((len(hist), start_token + len(hist) * tps, final_val))
    tail = [h[2] for h in hist[-20:]]
    res = {"label": label, "batch": batch, "seq": seq, "tokens_per_step": tps, "steps": len(hist),
           "start_token": start_token, "lr": lr, "warmup_steps": warmup, "decay_frac": decay_frac,
           "val_start": v0, "final_train": sum(tail) / len(tail), "final_val": final_val,
           "tok_per_s": n_timed * tps / max(t_train, 1e-9), "train_time_s": t_train,
           "wall_s": time.time() - t_start, "peak_gib": peak, "diverged": diverged,
           "hist": hist, "val_hist": val_hist, "loads": np.stack(loads) if loads else None}
    del opt
    gc.collect(); torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------------- diagnostics
def maxvio(counts):
    """(max load - mean load) / mean load over the last axis.

    Example: maxvio([2, 2, 2, 2]) -> 0.0;  maxvio([5, 1, 1, 1]) -> 1.5;  maxvio([8, 0, 0, 0]) -> 3.0
    """
    c = np.asarray(counts, dtype=float)
    m = c.mean(-1)
    return (c.max(-1) - m) / m


def dead_experts(counts, frac=0.01):
    """Number of experts whose load is below frac * the even load, over the last axis.

    Example: dead_experts([500, 500, 3, 997]) -> 1   (even load 500, threshold 5)
    """
    c = np.asarray(counts, dtype=float)
    return (c < frac * c.mean(-1, keepdims=True)).sum(-1)


@torch.no_grad()
def routing_probe(model, val, n_windows=64, batch=32):
    """Route n_windows validation windows and return (topi, ids, nxt): topi = list over MoE layers of the chosen
    experts (n_windows*T, k), ids = the routed input tokens (n_windows*T,), nxt = the next token of each."""
    was = model.training
    model.eval()
    moes = model.moe_layers()
    for m in moes:
        m.record = True
    tops, ids, nxt = [[] for _ in moes], [], []
    for i in range(0, n_windows, batch):
        w = val[i:min(i + batch, n_windows)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(w[:, :-1])
        for l, m in enumerate(moes):
            tops[l].append(m.last_topi.cpu())
        ids.append(w[:, :-1].reshape(-1).cpu()); nxt.append(w[:, 1:].reshape(-1).cpu())
    for m in moes:
        m.record = False
    model.train(was)
    return [torch.cat(t) for t in tops], torch.cat(ids), torch.cat(nxt)


@torch.no_grad()
def evaluate_capacity(model, val, cf, n_windows=1024, batch=64):
    """Validation loss and mean share of dropped assignments with capacity factor cf (0 = dropless).
    Capacity is per forward batch: ceil(batch * T * k / E * cf) assignments per expert per layer."""
    old = model.cfg.capacity_factor
    model.cfg.capacity_factor = cf
    was = model.training
    model.eval()
    try:
        tot, cnt, drops = 0.0, 0, []
        for i in range(0, n_windows, batch):
            w = val[i:min(i + batch, n_windows)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(w[:, :-1], w[:, 1:])
            tot += loss.item() * w.shape[0]; cnt += w.shape[0]
            drops.append(np.mean([m.last_dropped for m in model.moe_layers()]))
    finally:
        model.cfg.capacity_factor = old
        model.train(was)
    return tot / cnt, float(np.mean(drops))


@torch.no_grad()
def clone_similarity(model, family):
    """Per MoE layer, mean cosine similarity between experts of the same family (clones of one dense slice) and
    between experts of different families. Each expert is flattened to one vector [w1, w3, w2^T].
    Returns (same (L,), different (L,)) as numpy arrays; 1.0 = identical. Families of size 1 give nan for 'same'.
    Example: right after upcycle(..., "partition", r=0): same = 1.0 in every layer.
    """
    fam = torch.as_tensor(family)
    E = len(fam)
    same_mask = (fam[:, None] == fam[None, :]) & ~torch.eye(E, dtype=torch.bool)
    diff_mask = fam[:, None] != fam[None, :]
    same, diff = [], []
    for m in model.moe_layers():
        v = torch.cat([m.w1.flatten(1), m.w3.flatten(1), m.w2.transpose(1, 2).flatten(1)], 1).float()   # (E, 3*Fe*D)
        v = F.normalize(v, dim=1)
        S = (v @ v.T).cpu()                                                   # (E, E)
        same.append(S[same_mask].mean().item() if same_mask.any() else float("nan"))
        diff.append(S[diff_mask].mean().item() if diff_mask.any() else float("nan"))
    return np.array(same), np.array(diff)
