"""A small Llama-style decoder-only transformer, with an optional muP parameterization.

Architecture (same block design as Llama / Qwen / Mistral, at toy scale):

    token ids (B, T)
      -> Embedding                       (B, T, D)
      -> L x Block:
            x = x + Attention(RMSNorm(x))      causal self-attention with RoPE
            x = x + SwiGLU(RMSNorm(x))         gated feed-forward
      -> RMSNorm                          (B, T, D)
      -> LM head (separate matrix)        (B, T, V)   one score per vocabulary entry
      -> times 1/width_mult               (muP only; 1.0 in the standard parameterization)

Notation used in every docstring and shape comment:
    B = batch size (number of sequences)    T = sequence length (positions)
    D = d_model (width of the residual)     H = number of attention heads
    hd = D // H (width of one head)         F = d_ff (hidden width of the MLP)
    V = vocab_size                          L = n_layer

Differences from the Assignment 10 model:
  * the LM head is a separate matrix (no weight tying), so the embedding, the hidden matrices
    and the head can be given different learning rates, which muP needs;
  * hidden matrices are initialised at std 1/sqrt(fan_in) (the scale the Session 11 lesson
    assumes in its warmup argument), with the two residual-output projections further divided
    by sqrt(2L) so the residual stream does not grow with depth;
  * the head starts at std 1/D, so the first logits are tiny and the first loss is ln V.
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
    max_seq: int = 512        # longest sequence the RoPE tables are built for
    rope_base: float = 10000.0
    width_mult: float = 1.0   # muP: D / D_base. 1.0 = standard parameterization (no logit rescaling)


def swiglu_width(d_model, multiple=64):
    """SwiGLU hidden width: 8/3 * D rounded up to a multiple of 64.

    Example: swiglu_width(384) -> 1024,  swiglu_width(256) -> 704,  swiglu_width(1024) -> 2752
    """
    return multiple * math.ceil(8 * d_model / 3 / multiple)


class RMSNorm(nn.Module):
    """Root-mean-square normalisation: rescale each vector to unit RMS, then by a learned gain.

        y = x / sqrt(mean(x^2) + eps) * weight

    Computed in fp32 even under bf16 autocast, because squaring small bf16 numbers loses precision.

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
        xf = x.float()
        inv_rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * inv_rms).type_as(x) * self.weight


def rope_tables(cfg):
    """Precompute the cos/sin tables for rotary position embeddings (RoPE).

    Each pair of features (2i, 2i+1) inside a head is rotated by angle  pos * theta_i,
    with  theta_i = base^(-2i / hd).

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
        a' = a cos(phi) - b sin(phi),   b' = a sin(phi) + b cos(phi)
    so the dot product q_m . k_n depends only on the distance m - n.

    x: (B, H, T, hd)   cos, sin: (max_seq, hd/2)   ->   (B, H, T, hd)
    """
    T = x.shape[2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)     # (T, hd/2), broadcast over B and H
    a, b = x[..., 0::2], x[..., 1::2]                        # even and odd features, (B, H, T, hd/2)
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)   # (B, H, T, hd/2, 2)
    return rotated.flatten(-2)                               # interleave back to (B, H, T, hd)


class Attention(nn.Module):
    """Causal multi-head self-attention.

    Scores = q . k / sqrt(hd); softmax over earlier positions only; output = weighted sum of values.
    The head width hd is fixed at 64 across all model widths in this assignment, so the
    1/sqrt(hd) scale does not change with width (muP's 1/hd rule only matters when hd grows).

    x: (B, T, D) -> (B, T, D)
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)   # one matmul for q, k and v
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)      # mixes the heads back together

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x)                                                     # (B, T, 3D)
        # (B, T, D) -> (B, T, H, hd) -> (B, H, T, hd): heads become a batch dimension.
        q, k, v = (t.view(B, T, self.n_head, self.hd).transpose(1, 2) for t in qkv.split(D, dim=-1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # (B, H, T, hd)
        y = y.transpose(1, 2).reshape(B, T, D)                                # heads concatenated back
        return self.proj(y)


class SwiGLU(nn.Module):
    """Gated feed-forward network used by Llama.

        hidden = silu(x W_gate) * (x W_up)        (B, T, F)
        out    = hidden W_down                    (B, T, D)

    x: (B, T, D) -> (B, T, D)
    """

    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """One transformer layer: pre-norm attention and pre-norm MLP, each added to the residual.

    x: (B, T, D) -> (B, T, D).
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """The full decoder: token ids in, next-token scores (logits) out.

    Example:
        cfg = Config()                          # 49,255,296 weights (38.6M of them embedding + head)
        model = GPT(cfg)
        idx = torch.tensor([[464, 3290, 373]])  # (B=1, T=3) token ids
        logits = model(idx)                     # (1, 3, 50304)

    Initialisation (std of the normal distribution each matrix is drawn from):
        embedding                         0.02            (constant across widths, as muP wants)
        qkv, gate, up                     1/sqrt(fan_in)  e.g. 1/sqrt(384) = 0.0510
        attn.proj, mlp.down               1/sqrt(fan_in) / sqrt(2L)
        head                              1/D             so logits start at std 1/sqrt(D) ~ 0
        RMSNorm gains                     1.0
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=1.0 / self.cfg.d_model)
        depth_scale = 1.0 / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.blocks.named_parameters():
            if p.dim() == 2:                                   # nn.Linear weight: (fan_out, fan_in)
                std = 1.0 / math.sqrt(p.shape[1])
                if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                    std *= depth_scale
                nn.init.normal_(p, std=std)

    def forward(self, idx):
        """idx: (B, T) integer token ids  ->  logits: (B, T, V)."""
        x = self.embed(idx)                                    # (B, T, D)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)                     # (B, T, D)
        x = self.norm(x)
        # muP readout: dividing the logits by width_mult keeps their size, and the size of their
        # per-step change under Adam, independent of the width. In SP width_mult = 1 (no-op).
        return self.head(x) / self.cfg.width_mult              # (B, T, V)


def count_params(cfg):
    """Hand formula for the number of weights, to check against the real model.

        embedding V*D + head V*D + L * (attention 4*D^2 + MLP 3*D*F + two norms 2*D) + final norm D

    Example: Config() -> 2*50304*384 + 6*(4*384^2 + 3*384*1024 + 2*384) + 384 = 49,255,296
    """
    V, D, F_, L = cfg.vocab_size, cfg.d_model, cfg.d_ff, cfg.n_layer
    return 2 * V * D + L * (4 * D * D + 3 * D * F_ + 2 * D) + D
