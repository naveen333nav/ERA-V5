"""A small Llama-style decoder-only transformer.

Architecture (same block design as Llama / Qwen / Mistral, at toy scale):

    token ids (B, T)
      -> Embedding                      (B, T, D)
      -> L x Block:
            x = x + Attention(RMSNorm(x))      causal self-attention with RoPE
            x = x + SwiGLU(RMSNorm(x))         gated feed-forward
      -> RMSNorm                         (B, T, D)
      -> LM head (tied to the embedding) (B, T, V)   one score per vocabulary entry

Notation used in every docstring and shape comment:
    B = batch size (number of sequences)    T = sequence length (positions)
    D = d_model (width of the residual)     H = number of attention heads
    hd = D // H (width of one head)         F = d_ff (hidden width of the MLP)
    V = vocab_size                          L = n_layer

Shape tracing: pass `trace=[]` to `GPT.forward` and every intermediate tensor is
appended to it as (name, shape, dim_names, meaning). The notebook prints that list,
so the model code itself carries no print statements.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50304   # GPT-2 has 50,257 tokens; padded up to a multiple of 64 for faster matmuls
    n_layer: int = 6          # L
    n_head: int = 6           # H
    d_model: int = 384        # D  (so hd = 384 / 6 = 64)
    d_ff: int = 1024          # F  (SwiGLU hidden width, about 8/3 * D)
    max_seq: int = 512        # longest sequence RoPE tables are built for
    rope_base: float = 10000.0


def _at_least_fp32(dtype):
    """bf16/fp16 -> fp32, fp32 -> fp32, fp64 -> fp64. Upcast only, never downcast."""
    return torch.promote_types(dtype, torch.float32)


def _rec(trace, name, t, dims, meaning):
    """Append one entry to the shape trace, if tracing is on.

    Example:
        _rec(trace, "embed", x, ("B", "T", "D"), "one vector per token")
        -> trace gets ("embed", (2, 177, 384), ("B", "T", "D"), "one vector per token")
    """
    if trace is not None:
        trace.append((name, tuple(t.shape), dims, meaning))


class RMSNorm(nn.Module):
    """Root-mean-square normalisation: rescale each vector to unit RMS, then by a learned gain.

        y = x / sqrt(mean(x^2) + eps) * weight

    Unlike LayerNorm there is no mean subtraction and no bias, which is cheaper and works as well.
    Computed in at least fp32 even under bf16 autocast, because squaring small bf16 numbers loses precision.

    Example:
        x = [[3.0, 4.0]]          mean(x^2) = (9 + 16) / 2 = 12.5,  rms = 3.536
        RMSNorm(2)(x) = [[0.849, 1.131]]     (weight starts at 1)

    Input (..., D) -> output (..., D), same shape.
    """

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.to(_at_least_fp32(x.dtype))     # bf16 -> fp32, but fp64 stays fp64 (Step 2 needs that)
        inv_rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * inv_rms).type_as(x) * self.weight


def rope_tables(cfg):
    """Precompute the cos/sin tables for rotary position embeddings (RoPE).

    Each pair of features (2i, 2i+1) inside a head is rotated by angle  pos * theta_i,
    with  theta_i = base^(-2i / hd).  Low i rotates fast (fine position), high i slowly (coarse).

    Returns cos, sin, each of shape (max_seq, hd / 2).
    Example with hd = 4, base = 10000:  theta = [1.0, 0.01],
        position 3 -> angles [3.0, 0.03] -> cos [-0.990, 0.9996], sin [0.141, 0.030]
    """
    hd = cfg.d_model // cfg.n_head
    theta = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd))   # (hd/2,)
    angles = torch.outer(torch.arange(cfg.max_seq).float(), theta)           # (max_seq, hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate query or key vectors by their position.

    For each feature pair (a, b) at position p with angle phi = p * theta_i:
        a' = a cos(phi) - b sin(phi)
        b' = a sin(phi) + b cos(phi)
    After rotation, the dot product q_m . k_n depends only on the distance m - n,
    which is how attention learns relative position without a position embedding.

    x: (B, H, T, hd)   cos, sin: (max_seq, hd/2)   ->   (B, H, T, hd)
    """
    T = x.shape[2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)     # (T, hd/2), broadcast over B and H
    a, b = x[..., 0::2], x[..., 1::2]                        # even and odd features, (B, H, T, hd/2)
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)   # (B, H, T, hd/2, 2)
    return rotated.flatten(-2)                               # interleave back to (B, H, T, hd)


class Attention(nn.Module):
    """Causal multi-head self-attention.

    Each position builds a query ("what am I looking for"), a key ("what do I contain")
    and a value ("what I pass on"). Scores = q . k / sqrt(hd); softmax over earlier positions
    only (causal mask); output = weighted sum of values.

    x: (B, T, D) -> (B, T, D)
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)   # one matmul for q, k and v
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)      # mixes the heads back together

    def forward(self, x, cos, sin, trace=None, tag=""):
        B, T, D = x.shape
        qkv = self.qkv(x)
        _rec(trace, f"{tag}qkv", qkv, ("B", "T", "3D"), "query, key and value for every position, side by side")

        # (B, T, D) -> (B, T, H, hd) -> (B, H, T, hd): heads become a batch dimension,
        # so all H heads run as independent attentions in one batched matmul.
        q, k, v = (t.view(B, T, self.n_head, self.hd).transpose(1, 2) for t in qkv.split(D, dim=-1))
        _rec(trace, f"{tag}q (per head)", q, ("B", "H", "T", "hd"), "query vectors, split into H heads of width hd")

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if trace is not None:
            # Only for the printout. The fused kernel below never builds this matrix in memory,
            # which is exactly why it saves memory: this tensor grows as T^2.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            _rec(trace, f"{tag}scores", scores, ("B", "H", "Tq", "Tk"),
                 "how much each query position attends to each key position")

        # Right-padded batches need no padding mask: the causal mask already stops a real token
        # from attending to positions after it, and all padding sits after the real tokens.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # (B, H, T, hd)
        y = y.transpose(1, 2).reshape(B, T, D)                                # heads concatenated back
        _rec(trace, f"{tag}attn out", y, ("B", "T", "D"), "attention result per position, heads concatenated")
        return self.proj(y)


class SwiGLU(nn.Module):
    """Gated feed-forward network used by Llama.

        hidden = silu(x W_gate) * (x W_up)        (B, T, F)    the gate decides how much of "up" passes
        out    = hidden W_down                    (B, T, D)

    Three matrices instead of the classic two, so F is about 8/3 * D (not 4 * D) to keep the
    weight count similar.

    x: (B, T, D) -> (B, T, D)
    """

    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, trace=None, tag=""):
        h = F.silu(self.gate(x)) * self.up(x)
        _rec(trace, f"{tag}mlp hidden", h, ("B", "T", "F"), "widened per-token features inside the MLP")
        return self.down(h)


class Block(nn.Module):
    """One transformer layer: pre-norm attention and pre-norm MLP, each added to the residual.

    x: (B, T, D) -> (B, T, D). The residual stream keeps its shape through every layer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, trace=None, tag=""):
        x = x + self.attn(self.norm1(x), cos, sin, trace, tag)
        x = x + self.mlp(self.norm2(x), trace, tag)
        _rec(trace, f"{tag}block out", x, ("B", "T", "D"), "residual stream after this layer")
        return x


class GPT(nn.Module):
    """The full decoder: token ids in, next-token scores (logits) out.

    Example:
        cfg = Config()                          # 29,938,560 weights
        model = GPT(cfg)
        idx = torch.tensor([[464, 3290, 373]])  # (B=1, T=3) token ids
        logits = model(idx)                     # (1, 3, 50304)
        logits[0, -1].argmax()                  # id of the model's guess for token 4

    The LM head shares its matrix with the embedding (weight tying): the same V x D matrix
    turns ids into vectors on the way in and vectors into scores on the way out.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Small random weights (std 0.02, as in GPT-2): logits start near zero, so the first
        # prediction is close to uniform and the initial loss is close to ln(V).
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, trace=None):
        """idx: (B, T) integer token ids  ->  logits: (B, T, V)."""
        _rec(trace, "tokens", idx, ("B", "T"), "input token ids")
        x = self.embed(idx)
        _rec(trace, "embed", x, ("B", "T", "D"), "one D-dim vector per token (row lookup in the embedding)")
        for i, blk in enumerate(self.blocks):
            # Trace the first layer only; the other L-1 layers produce identical shapes.
            x = blk(x, self.cos, self.sin, trace if i == 0 else None, "block0 ")
        x = self.norm(x)
        logits = self.head(x)
        _rec(trace, "logits", logits, ("B", "T", "V"), "a score for every vocabulary entry, at every position")
        return logits


def loss_sum_and_count(logits, targets):
    """Summed cross-entropy over real (non-padding) tokens, and how many real tokens there were.

    logits:  (B, T, V)   targets: (B, T) with -100 at padding positions
    returns: (loss_sum: scalar fp32 tensor, n_tokens: scalar int tensor)

    Example: targets = [[6, 7], [9, -100]] -> 3 real tokens, so loss_sum adds 3 per-token
    losses and n_tokens = 3. The mean is loss_sum / n_tokens.

    The sum and the count are returned separately, not as their ratio, so that a caller
    accumulating several micro-batches can divide by the TOTAL token count of the whole batch.
    That is the fix for the gradient-accumulation bug studied in Step 3.
    """
    loss_sum = F.cross_entropy(
        logits.to(_at_least_fp32(logits.dtype)).view(-1, logits.size(-1)),   # (B*T, V); log-softmax over 50k wants >= fp32
        targets.view(-1),                           # (B*T,)
        ignore_index=-100, reduction="sum",
    )
    return loss_sum, (targets != -100).sum()
