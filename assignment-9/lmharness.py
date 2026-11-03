"""Llama-style decoder LM and an observable next-token loss harness (ERA V5, Session 9).

Architecture (pre-norm Llama block, the same family as Assignments 10-14):

    token ids (B, T)
      -> Embedding E (V, D)                                             x (B, T, D), fp32 residual stream
      -> L blocks:  x = x + Attn(RMSNorm(x))          causal, optional document mask, RoPE
                    x = x + SwiGLU(RMSNorm(x))
      -> final RMSNorm                                                  hidden (B, T, D)
      -> head 1:  logits  = hidden @ W1^T,  W1 = E (tied) or its own (V, D) matrix      predicts token t+1
      -> head 2 (optional), predicts token t+2 from position t:
            "dense":      logits2 = h_t @ W2^T with its own (V, D) matrix                (reads h_t only)
            "cheap":      logits2 = RMSNorm(h_t + SwiGLU2(RMSNorm(h_t))) @ W1^T          (reads h_t only)
            "sequential": u_t = Merge([RMSNorm(h_t); RMSNorm(E[x_{t+1}])])  (2D -> D), one causal Block over u,
                          logits2 = RMSNorm(u_t) @ W1^T                  (DeepSeek-V3 style: also reads the true
                                                                          token t+1 in training, head 1's guess at
                                                                          inference)

The loss harness (everything between the hidden state and the scalar loss):

    targets_k[t] = tokens[t + k]   if positions t and t + k are in the same document and neither is padding
                 = IGNORE (-100)   otherwise
    loss_k       = sum over valid t of  -log softmax(logits_k[t])[targets_k[t]]   /   number of valid t

ChunkedCrossEntropy computes loss_1 from (hidden, W1) without ever holding the (N, V) logits: it walks over the rows
in chunks, keeps one log-normaliser per row, and recomputes each chunk's logits in the backward pass.

Notation in docstrings and shape comments:
    B sequences per batch   T tokens per sequence   D d_model   H heads   hd = D / H   F FFN width
    V vocabulary size       L layers                N rows given to the loss (B * (T - 1) for head 1)
"""
import gc
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EOT = 50256          # GPT-2 <|endoftext|>: closes every document
PAD = 50257          # our own padding id (GPT-2 has none); never appears in packed training data
IGNORE = -100        # target value that F.cross_entropy(ignore_index=IGNORE) skips


@dataclass
class Config:
    vocab_size: int = 50304        # V: GPT-2's 50,257 ids + PAD, rounded up to a multiple of 64 (46 rows never used)
    n_layer: int = 6               # L
    n_head: int = 6                # H   (hd = 384 / 6 = 64)
    d_model: int = 384             # D
    d_ff: int = 1024               # F = 8/3 * D, the SwiGLU width with the same budget as a 4x two-matrix FFN
    max_seq: int = 512             # T the RoPE tables are built for
    rope_base: float = 10000.0
    tie: bool = True               # head 1 reuses the embedding matrix
    head2: str = "none"            # "none" | "dense" | "cheap" | "sequential"
    init_std: float = 0.02         # std of every weight matrix (residual output projections get / sqrt(2L))
    head_init_std: float = 0.02    # std of an untied head 1 and of a dense head 2
    seed: int = 0                  # init seed; embedding and trunk come first, so every config shares them


# ----------------------------------------------------------------------------------------- targets
def doc_ids(tokens, eot=EOT):
    """Document index of every position; an <eot> belongs to the document it closes.

    Example: doc_ids(torch.tensor([[5, 6, EOT, 7, 8]])) -> tensor([[0, 0, 0, 1, 1]])
    """
    is_eot = (tokens == eot).long()                                         # (B, T)
    return torch.cumsum(is_eot, dim=1) - is_eot                             # number of <eot> strictly before t


def make_targets(tokens, k=1, *, mask_pad=True, mask_boundary=True, eot=EOT, pad=PAD):
    """Targets for logits[:, :-k]: target[t] = tokens[t + k], or IGNORE where that pair must not count. (B, T-k).

    mask_boundary: ignore the pair if t and t + k sit in different documents (the next document is unrelated).
    mask_pad:      ignore the pair if the input or the target is padding.
    Example (k=1): tokens [[5, 6, EOT, 7, 8, PAD]] -> [[6, EOT, IGNORE, 8, IGNORE]]
        6 after 5: kept;  EOT after 6: kept (ending a document is a real prediction);
        7 after EOT: first token of a new document -> IGNORE;  8 after 7: kept;  PAD after 8 -> IGNORE.
    Example (k=2): tokens [[5, 6, EOT, 7, 8, 9]] -> [[EOT, IGNORE, IGNORE, 9]]
        (6 -> 7 and EOT -> 8 both jump into the next document; 7 -> 9 stays inside it)
    """
    tgt = tokens[:, k:].clone()                                             # (B, T-k)
    if mask_boundary:
        d = doc_ids(tokens, eot)
        tgt[d[:, k:] != d[:, :-k]] = IGNORE
    if mask_pad:
        tgt[(tokens[:, k:] == pad) | (tokens[:, :-k] == pad)] = IGNORE
    return tgt


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
    inv = base ** (-torch.arange(0, hd, 2).float() / hd)                    # (hd/2,)
    ang = torch.arange(T).float()[:, None] * inv[None]                      # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate each (even, odd) feature pair of x (B, H, T, hd) by its position's angle. Norm-preserving.

    Example: one head, hd=2, position 1, angle 1 rad: (1, 0) -> (cos 1, sin 1) = (0.5403, 0.8415)
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]                                     # (B, H, T, hd/2) each
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)   # (B, H, T, hd/2, 2)
    return out.flatten(-2)                                                  # (B, H, T, hd)


def attention_mask(doc, causal=True):
    """Boolean mask (B, 1, T, T): True where query t may attend to key s (same document, and s <= t if causal).

    Example: doc [[0, 0, 1]], causal -> [[1, 0, 0], [1, 1, 0], [0, 0, 1]]  (token 2 cannot see document 0)
    """
    T = doc.shape[1]
    m = doc[:, None, :, None] == doc[:, None, None, :]                      # (B, 1, T, T) same document
    if causal:
        m = m & torch.ones(T, T, dtype=torch.bool, device=doc.device).tril()
    return m


class Attention(nn.Module):
    """Multi-head self-attention with RoPE, no biases. (B, T, D) -> (B, T, D).

    mask: None -> plain causal (or full, if causal=False) attention; else a (B, 1, T, T) boolean mask.
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

    def forward(self, x, mask=None, causal=True, trace=None):
        B, T, D = x.shape
        qkv = self.qkv(x)                                                   # (B, T, 3D)
        q, k, v = qkv.view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4) # 3 x (B, H, T, hd)
        cos, sin = self.cos[:T].to(q.dtype), self.sin[:T].to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)  # (B, H, T, hd)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)    # (B, H, T, hd)
        y = y.transpose(1, 2).reshape(B, T, D)                              # (B, T, D) heads side by side
        out = self.proj(y)                                                  # (B, T, D)
        if trace:
            trace("block 0  qkv = Linear(D, 3D)", qkv, "(B, T, 3D): query, key and value for every token, packed")
            trace("block 0  q (also k, v), after RoPE", q, "(B, H, T, hd): split into H heads of width hd = D/H")
            trace("block 0  attention output per head", y.view(B, T, self.H, self.hd).transpose(1, 2),
                  "(B, H, T, hd): each head's weighted mix of value vectors")
            trace("block 0  heads merged + out projection", out, "(B, T, D): back to model width, added to x")
        return out


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)): three matrices, width F. (B, T, D) -> (B, T, D).

    Example: SwiGLU(Config())(torch.zeros(1, 1, 384)) -> zeros (1, 1, 384)   (silu(0) = 0)
    """

    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, trace=None):
        a = F.silu(self.gate(x)) * self.up(x)                               # (B, T, F)
        if trace:
            trace("block 0  FFN silu(gate) * up", a, "(B, T, F): the wide middle of the feed-forward network")
        return self.down(a)                                                 # (B, T, D)


class Block(nn.Module):
    """Pre-norm block: x = x + Attn(norm(x)); x = x + FFN(norm(x)). (B, T, D) -> (B, T, D)."""

    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, mask=None, causal=True, trace=None):
        x = x + self.attn(self.n1(x), mask, causal, trace)
        if trace:
            trace("block 0  x after attention residual", x, "(B, T, D): residual stream, attention result added")
        x = x + self.ffn(self.n2(x), trace)
        if trace:
            trace("block 0  x after FFN residual", x, "(B, T, D): residual stream leaving block 0")
        return x


class LM(nn.Module):
    """Decoder-only LM split the way the assignment's harness is written:

        hidden  = model(tokens)            (B, T) -> (B, T, D)
        logits  = model.head(hidden)       (B, T, D) -> (B, T, V)      predicts t+1
        logits2 = model.head2(hidden, tokens)   -> (B, T-2, V)         predicts t+2 (cfg.head2 != "none"),
                                                                       already aligned with make_targets(tokens, 2)

    Example: m = LM(Config()); h = m(torch.zeros(2, 8, dtype=torch.long)); m.head(h).shape -> (2, 8, 50304)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie:
            self.lm_head.weight = self.emb.weight                           # one (V, D) tensor, two uses
        if cfg.head2 == "dense":
            self.head2_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        elif cfg.head2 == "cheap":
            self.head2_norm_in, self.head2_ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)
            self.head2_norm_out = RMSNorm(cfg.d_model)
        elif cfg.head2 == "sequential":
            self.head2_norm_h, self.head2_norm_e = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
            self.head2_merge = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
            self.head2_block = Block(cfg)
            self.head2_norm_out = RMSNorm(cfg.d_model)
        elif cfg.head2 != "none":
            raise ValueError(f"unknown head2 {cfg.head2!r}")
        self._init()

    @torch.no_grad()
    def _init(self):
        """Normal(0, init_std) for every matrix; residual output projections / sqrt(2L); untied heads head_init_std.

        One generator seeded with cfg.seed, drawn in registration order (emb, blocks, norm, lm_head, head2):
        two configs that differ only in their heads get the identical embedding and trunk.
        """
        c = self.cfg
        g = torch.Generator().manual_seed(c.seed)
        for name, p in self.named_parameters():                           # a tied matrix appears once, as emb.weight
            if p.dim() < 2:
                continue                                                    # norm gains stay at 1
            std = c.init_std
            if name.endswith("proj.weight") and "attn" in name or name.endswith("down.weight"):
                std = c.init_std / math.sqrt(2 * c.n_layer)
            if name.startswith("lm_head") or name.startswith("head2_proj"):
                std = c.head_init_std
            p.normal_(0.0, std, generator=g)

    @property
    def has_head2(self):
        return self.cfg.head2 != "none"

    def forward(self, idx, doc=None, causal=True, trace=None):
        """idx (B, T) token ids -> hidden (B, T, D). doc (B, T) document ids switches on document-aware attention."""
        if trace:
            trace("tokens", idx, "(B, T): B sequences in the batch, T token positions per sequence")
        x = self.emb(idx)                                                   # (B, T, D)
        if trace:
            trace("x = embedding(tokens)", x, "(B, T, D): one D-wide vector per token, the residual stream")
        mask = attention_mask(doc, causal) if doc is not None else None     # (B, 1, T, T) or None
        if trace and mask is not None:
            trace("attention mask (same doc, causal)", mask, "(B, 1, T, T): query t may see key s; 1 = shared by heads")
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, causal, trace if i == 0 else None)
        h = self.norm(x)                                                    # (B, T, D)
        if trace:
            trace(f"hidden = RMSNorm(x after {len(self.blocks)} blocks)", h,
                  "(B, T, D): the hidden state the output head reads")
        return h

    def head(self, hidden):
        """(..., D) -> (..., V) logits for token t+1."""
        return self.lm_head(hidden)

    def head2(self, hidden, tokens):
        """Logits for token t+2 made at positions t = 0..T-3. hidden (B, T, D), tokens (B, T) -> (B, T-2, V).

        The output is already aligned with make_targets(tokens, 2): row t is scored against tokens[:, t + 2].
        Only the "sequential" head reads `tokens`: at position t it also sees the embedding of tokens[:, t + 1]
        (the true token in training; at inference that slot holds head 1's own guess).
        Example: dense head, hidden (2, 8, 384) -> (2, 6, 50304)
        """
        h = hidden[:, :-2]                                                  # (B, T-2, D) positions 0..T-3
        c = self.cfg.head2
        if c == "dense":
            return self.head2_proj(h)
        if c == "cheap":
            u = h + self.head2_ffn(self.head2_norm_in(h))                   # (B, T-2, D) one extra FFN step
            return F.linear(self.head2_norm_out(u), self.lm_head.weight)    # shared unembedding
        if c == "sequential":
            e_next = self.emb(tokens[:, 1:-1])                              # (B, T-2, D) embedding of token t+1
            u = self.head2_merge(torch.cat([self.head2_norm_h(h), self.head2_norm_e(e_next)], dim=-1))  # (B, T-2, D)
            u = self.head2_block(u, attention_mask(doc_ids(tokens[:, :-2])))  # causal, same-document attention
            return F.linear(self.head2_norm_out(u), self.lm_head.weight)    # shared unembedding
        raise RuntimeError("this model has no second head")


def count_params(model):
    """Parameter counts. Example: count_params(LM(Config()))['total'] -> 29,938,560 (tied, one head)."""
    c = model.cfg
    total = sum(p.numel() for p in model.parameters())                      # parameters() lists a shared tensor once
    sd_total = sum(t.numel() for t in model.state_dict().values())           # state_dict() lists it under both names
    emb = model.emb.weight.numel()
    head1 = 0 if c.tie else model.lm_head.weight.numel()
    head2 = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head2"))
    return {"total": total, "state_dict_total": sd_total, "embedding": emb, "head1_own": head1,
            "head2": head2, "trunk": total - emb - head1 - head2}


# ----------------------------------------------------------------------------------------- the harness
def harness_losses(model, tokens, bug="none"):
    """The Session 9 harness on one batch, plus the deliberate bugs of Part 3. tokens (B, T) -> dict of losses.

    bug = "none"           correct: logits[:, :-1] against make_targets(tokens, 1); head 2 against tokens t+2
          "no_shift"       targets = inputs: each position is asked for the token it already holds
          "reverse_shift"  logits[:, 1:] against tokens[:, :-1]: each position is asked for the token before it
          "no_causal"      correct targets, but attention may look at later tokens
          "count_pad"      padded rows, pad targets counted ("real1" = the same batch's loss on real targets only)
    Returns {"loss1": scalar, ["loss2": scalar], ["real1": scalar], "total": loss1 + loss2}
    """
    doc = doc_ids(tokens)
    hidden = model(tokens, doc=doc, causal=(bug != "no_causal"))           # (B, T, D)
    logits = model.head(hidden)                                             # (B, T, V)
    V = logits.shape[-1]
    out = {}
    if bug == "no_shift":
        tgt = tokens.masked_fill(tokens == PAD, IGNORE)                     # (B, T)
        out["loss1"] = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=IGNORE)
    elif bug == "reverse_shift":
        tgt = tokens[:, :-1].masked_fill(tokens[:, :-1] == PAD, IGNORE)     # (B, T-1)
        out["loss1"] = F.cross_entropy(logits[:, 1:].reshape(-1, V), tgt.reshape(-1), ignore_index=IGNORE)
    elif bug == "count_pad":
        z = logits[:, :-1].reshape(-1, V)                                   # (B*(T-1), V)
        t_all = make_targets(tokens, 1, mask_pad=False, mask_boundary=False).reshape(-1)
        out["loss1"] = F.cross_entropy(z, t_all, ignore_index=IGNORE)
        with torch.no_grad():
            out["real1"] = F.cross_entropy(z, make_targets(tokens, 1).reshape(-1), ignore_index=IGNORE)
    elif bug in ("none", "no_causal"):
        out["loss1"] = F.cross_entropy(logits[:, :-1].reshape(-1, V), make_targets(tokens, 1).reshape(-1),
                                       ignore_index=IGNORE)
    else:
        raise ValueError(f"unknown bug {bug!r}")
    if model.has_head2 and bug == "none":
        logits2 = model.head2(hidden, tokens)                               # (B, T-2, V), already shifted by 2
        out["loss2"] = F.cross_entropy(logits2.reshape(-1, V), make_targets(tokens, 2).reshape(-1),
                                       ignore_index=IGNORE)
    out["total"] = out["loss1"] + out["loss2"] if "loss2" in out else out["loss1"]
    return out


@torch.no_grad()
def evaluate(model, windows, batch=32, doc_mask=True, mask_boundary=True):
    """Token-weighted mean loss and top-1 accuracy of head 1 (and head 2) over windows (n, T), correct harness.

    Returns e.g. {'loss1': 2.41, 'acc1': 0.47, 'n1': 261632, 'loss2': 3.9, 'acc2': 0.27, 'n2': 261120}
    """
    was_training = model.training
    model.eval()
    ks = [1, 2] if model.has_head2 else [1]
    sums = {k: [0.0, 0, 0] for k in ks}                                     # loss sum, correct, count
    for s in range(0, len(windows), batch):
        tok = windows[s:s + batch].long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(tok, doc=doc_ids(tok) if doc_mask else None)          # (b, T, D)
            for k in ks:
                z = model.head(h[:, :-1]) if k == 1 else model.head2(h, tok)  # (b, T-k, V)
                tgt = make_targets(tok, k, mask_boundary=mask_boundary)     # (b, T-k)
                valid = tgt != IGNORE
                sums[k][0] += F.cross_entropy(z.reshape(-1, z.shape[-1]).float(), tgt.reshape(-1),
                                              ignore_index=IGNORE, reduction="sum").item()
                sums[k][1] += ((z.argmax(-1) == tgt) & valid).sum().item()
                sums[k][2] += valid.sum().item()
                del z
    model.train(was_training)
    out = {}
    for k, (ls, corr, n) in sums.items():
        out[f"loss{k}"], out[f"acc{k}"], out[f"n{k}"] = ls / n, corr / n, n
    return out


@torch.no_grad()
def per_token_losses(model, tokens, k=1, doc_mask=True, mask_boundary=True, mask_pad=True):
    """Loss of every target position, NaN where ignored. tokens (B, T) -> (B, T-k) float32 (bf16 autocast forward).

    Example: per_token_losses(untrained_model, tok)[0, :3] -> tensor([10.91, 10.87, 10.93])  (about ln V each)
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(tokens, doc=doc_ids(tokens) if doc_mask else None)
        z = (model.head(h[:, :-1]) if k == 1 else model.head2(h, tokens)).float()   # (B, T-k, V)
    tgt = make_targets(tokens, k, mask_pad=mask_pad, mask_boundary=mask_boundary)
    l = F.cross_entropy(z.reshape(-1, z.shape[-1]), tgt.reshape(-1), ignore_index=IGNORE, reduction="none")
    return l.view(tgt.shape).masked_fill(tgt == IGNORE, float("nan"))


# ----------------------------------------------------------------------------------------- chunked cross-entropy
class ChunkedCrossEntropy(torch.autograd.Function):
    """mean over valid rows of  logsumexp(h W^T) - (h W^T)[target],  without materialising the (N, V) logits.

    forward:  for each chunk of rows, z = h_c W^T (c, V); keep lse = logsumexp(z) (c floats) and the summed loss;
              z is thrown away before the next chunk.
    backward: recompute z for each chunk; dL/dz = (softmax(z) - onehot(y)) / n_valid = (exp(z - lse) - onehot) / n;
              dL/dh_c = dz W (c, D);   dL/dW += dz^T h_c (V, D).
    Peak extra memory ~ a few chunk x V tensors instead of a few N x V tensors.
    Under bf16 autocast the inputs are cast to bf16 (like F.linear would) and the softmax maths runs in fp32.

    Example: h (8176, 384), W (50304, 384), y (8176,) -> same scalar as F.cross_entropy(h @ W.T, y), to ~1e-6.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, h, W, targets, chunk):
        N = h.shape[0]
        acc = torch.float64 if h.dtype == torch.float64 else torch.float32  # accumulation dtype of the softmax maths
        valid = targets != IGNORE                                           # (N,)
        safe_t = targets.clamp(min=0)                                       # ignored rows point at column 0
        lse = torch.empty(N, dtype=acc, device=h.device)                    # (N,) one log-normaliser per row
        total = torch.zeros((), dtype=acc, device=h.device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            z = (h[s:e] @ W.t()).to(acc)                                    # (c, V) this chunk's logits only
            lse[s:e] = torch.logsumexp(z, dim=-1)
            zy = z.gather(1, safe_t[s:e, None]).squeeze(1)                  # (c,) logit of the true token
            total += ((lse[s:e] - zy) * valid[s:e]).sum()
            del z
        n_valid = valid.sum().clamp(min=1)
        ctx.save_for_backward(h, W, safe_t, valid, lse, n_valid)
        ctx.chunk = chunk
        return total / n_valid

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, g_out):
        h, W, safe_t, valid, lse, n_valid = ctx.saved_tensors
        acc = lse.dtype
        scale = g_out.to(acc) / n_valid                                     # d(mean)/d(row loss)
        gh = torch.empty(h.shape, dtype=acc, device=h.device)               # (N, D)
        gW = torch.zeros(W.shape, dtype=acc, device=W.device)               # (V, D), accumulated over chunks in fp32
        for s in range(0, h.shape[0], ctx.chunk):
            e = min(s + ctx.chunk, h.shape[0])
            hc = h[s:e]
            p = (hc @ W.t()).to(acc)                                        # (c, V) recomputed logits
            p.sub_(lse[s:e, None]).exp_()                                   # softmax, in place
            p[torch.arange(e - s, device=p.device), safe_t[s:e]] -= 1.0     # softmax - onehot
            p.mul_((valid[s:e].to(acc) * scale)[:, None])                   # ignored rows -> 0, divide by n_valid
            gz = p.to(h.dtype)
            del p
            gh[s:e] = gz @ W                                                # (c, D)
            gW += (gz.t() @ hc).to(acc)                                     # (V, D)
            del gz
        # Returned in fp32 (fp64 under gradcheck); autograd casts them to the dtype of the original inputs.
        return gh, gW, None, None


def chunked_cross_entropy(h, W, targets, chunk=1024):
    """h (N, D), W (V, D), targets (N,) with IGNORE allowed -> scalar mean loss. See ChunkedCrossEntropy."""
    return ChunkedCrossEntropy.apply(h, W, targets, chunk)


def ordinary_cross_entropy(h, W, targets):
    """The materialising reference: logits = h W^T (N, V), then F.cross_entropy. Same inputs and output."""
    return F.cross_entropy(F.linear(h, W), targets, ignore_index=IGNORE)


# ----------------------------------------------------------------------------------------- training
def make_optimizer(model, lr, wd=0.1):
    """AdamW(0.9, 0.95); weight decay on matrices only (norm gains are not decayed)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), fused=True)


def lr_at(step, total, peak, warmup, floor=0.1):
    """Linear warm-up for `warmup` steps, then cosine from peak to floor * peak at the last step.

    Example: lr_at(0, 2000, 1e-3, 100) -> 1e-05;  lr_at(99, 2000, 1e-3, 100) -> 1e-03;  lr_at(1999, ...) -> 1e-04
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - 1 - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))


def train(model, *, steps, data=None, rows=None, val=None, batch=16, seq=512, lr=1e-3, warmup=100, eval_every=100,
          bug="none", label="run", log_every=200, seed=1337):
    """Train `model` in place for `steps` optimizer steps with the harness of harness_losses(bug=...).

    data: flat int32 token stream on the GPU (packed documents): step s reads data[s*B*T : (s+1)*B*T] as (B, T).
    rows: (n, T) padded rows on the GPU (bug="count_pad"): step s reads rows[s*B : (s+1)*B].
    val:  (n, T) validation windows, scored with the correct harness every eval_every steps (and at step 0).
    Loss = loss1 (+ loss2 for a two-head model), each a mean over its own valid targets; bf16 autocast; clip 1.0.

    Returns {'label', 'steps', 'hist': [(step, loss1, loss2, real1, grad_norm, lr)], 'val_hist': [(step, eval)],
             'final': evaluate(...), 'tok_per_s', 'peak_gib', 'train_time_s'}  (loss2/real1 are NaN when absent)
    """
    torch.manual_seed(seed)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = make_optimizer(model, lr)
    tps = batch * seq
    hist, val_hist = [], [(0, evaluate(model, val))]
    t_train, n_timed = 0.0, 0
    nan = float("nan")
    print(f"  [{label}] {steps} steps x {tps:,} tokens; val loss before training: "
          + ", ".join(f"head {k[-1]} {v:.4f}" for k, v in val_hist[0][1].items() if k.startswith("loss")))
    for step in range(steps):
        t0 = time.perf_counter()
        if rows is None:
            tok = data[step * tps:(step + 1) * tps].view(batch, seq).long()
        else:
            tok = rows[step * batch:(step + 1) * batch].long()
        cur_lr = lr_at(step, steps, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = harness_losses(model, tok, bug=bug)
        out["total"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        l1 = out["loss1"].item()                                            # synchronises the GPU
        if step >= 5:
            t_train += time.perf_counter() - t0
            n_timed += 1
        hist.append((step + 1, l1, out["loss2"].item() if "loss2" in out else nan,
                     out["real1"].item() if "real1" in out else nan, float(gnorm), cur_lr))
        if not math.isfinite(l1):
            print(f"  [{label}] loss became {l1} at step {step + 1}: stopping")
            break
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            val_hist.append((step + 1, evaluate(model, val)))
        if (step + 1) % log_every == 0 or step == 0 or step + 1 == steps:
            h = hist[-1]
            extra = (f"  loss2 {h[2]:.4f}  sum {h[1] + h[2]:.4f}" if math.isfinite(h[2]) else "") + \
                    (f"  real-token loss {h[3]:.4f}" if math.isfinite(h[3]) else "")
            vt = ""
            if val_hist[-1][0] == step + 1:
                ev = val_hist[-1][1]
                vt = f"  | val head 1 {ev['loss1']:.4f}" + (f", head 2 {ev['loss2']:.4f}" if "loss2" in ev else "")
            print(f"  [{label}] step {step + 1:5d}/{steps}  train loss1 {h[1]:.4f}{extra}  "
                  f"grad norm {h[4]:.3f}  lr {cur_lr:.2e}{vt}")
    res = {"label": label, "bug": bug, "steps": len(hist), "tokens_per_step": tps, "lr": lr, "warmup": warmup,
           "hist": hist, "val_hist": val_hist, "final": val_hist[-1][1],
           "tok_per_s": n_timed * tps / max(t_train, 1e-9), "train_time_s": t_train,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    del opt
    gc.collect(); torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, prompt, n_new, causal=True):
    """Greedy decoding with head 1, full recompute each step (no KV cache). prompt (B, T0) -> (B, T0 + n_new)."""
    x = prompt.long()
    for _ in range(n_new):
        ctx = x[:, -model.cfg.max_seq:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(ctx, doc=doc_ids(ctx), causal=causal)
            nxt = model.head(h[:, -1:]).argmax(-1)                          # (B, 1)
        x = torch.cat([x, nxt], dim=1)
    return x


@torch.no_grad()
def draft_acceptance(model, prompts, n_new):
    """Greedy self-speculative decoding with head 2 as the drafter, scored exactly from one greedy continuation.

    At step t head 1 emits g[t+1] and head 2 drafts a guess for g[t+2]. The next pass accepts the draft iff
    head 1, having read g[t+1], also picks it, i.e. iff draft == g[t+2] in the greedy sequence.
    A "sequential" head 2 reads g[t+1] itself, which is exactly what it would see at inference.
    Returns (acceptance rate over all drafts, generated sequences (B, T0 + n_new)).
    """
    seq = generate(model, prompts, n_new)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(seq, doc=doc_ids(seq))
        draft = model.head2(h, seq).argmax(-1)                              # (B, T-2): guess for token t+2
    T0, T = prompts.shape[1], seq.shape[1]
    t = torch.arange(T0 - 1, T - 2, device=seq.device)                       # drafts whose target was generated
    return (draft[:, t] == seq[:, t + 2]).float().mean().item(), seq
