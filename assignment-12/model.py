"""A small Llama-style decoder (RMSNorm, RoPE, SwiGLU) laid out as a list of "units" for sharding.

This is the Assignment 10/11 decoder, reorganised so that ZeRO can treat it piece by piece:

    token ids (B, T)
      -> units[0]      Embed                         (B, T, D)
      -> units[1..L]   Block  (one transformer layer) (B, T, D)
            x = x + Attention(RMSNorm(x))      causal self-attention with RoPE
            x = x + SwiGLU(RMSNorm(x))         gated feed-forward
      -> units[L+1]    Head   (final RMSNorm + LM head) (B, T, V)

A *unit* is the granularity at which the ZeRO engine reduces gradients (ZeRO-2) and gathers weights
(ZeRO-3). PyTorch FSDP calls this the "wrapping" granularity; DeepSpeed calls the gradient version a bucket.

Notation used in every docstring and shape comment:
    B = batch size (sequences)    T = sequence length     D = d_model (residual width)
    H = attention heads           hd = D // H             F = d_ff (SwiGLU hidden width)
    V = vocab_size                L = n_layer

Every module works in fp64, fp32 or bf16: RMSNorm only upcasts 16-bit inputs, so the fp64 equivalence runs
in the notebook stay in fp64 end to end.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096    # GPT-2 tokens remapped to the 4,095 most frequent TinyStories tokens + 1 "rare" token
    n_layer: int = 4          # L
    n_head: int = 4           # H
    d_model: int = 256        # D  (so hd = 64)
    d_ff: int = 704           # F  (8/3 * D rounded up to a multiple of 64)
    max_seq: int = 256        # longest sequence the RoPE tables are built for
    rope_base: float = 10000.0


def upcast(x):
    """16-bit tensors -> fp32; fp32 and fp64 are left alone.  Example: upcast(bf16 tensor).dtype -> torch.float32"""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class RMSNorm(nn.Module):
    """Root-mean-square normalisation:  y = x / sqrt(mean(x^2) + eps) * weight.

    Example:  x = [[3.0, 4.0]]  ->  mean(x^2) = 12.5, rms = 3.536  ->  RMSNorm(2)(x) = [[0.849, 1.131]]
    Input (..., D) -> output (..., D).
    """

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = upcast(x)
        inv_rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * inv_rms).type_as(x) * self.weight


def rope_tables(cfg):
    """cos/sin tables for rotary position embeddings, each (max_seq, hd/2).

    Feature pair i of a head is rotated by angle pos * theta_i with theta_i = base^(-2i/hd).
    Example (hd = 4): theta = [1.0, 0.01]; position 3 -> angles [3.0, 0.03] -> cos [-0.990, 0.9996]
    """
    hd = cfg.d_model // cfg.n_head
    theta = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2).double() / hd))     # (hd/2,)
    angles = torch.outer(torch.arange(cfg.max_seq).double(), theta)             # (max_seq, hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each (even, odd) feature pair of q or k by its position.  x: (B, H, T, hd) -> (B, H, T, hd)."""
    T = x.shape[2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)                         # (T, hd/2)
    a, b = x[..., 0::2], x[..., 1::2]                                            # (B, H, T, hd/2) each
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Causal multi-head self-attention with RoPE.  x: (B, T, D) -> (B, T, D)."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x)                                                         # (B, T, 3D)
        q, k, v = (t.view(B, T, self.n_head, self.hd).transpose(1, 2) for t in qkv.split(D, dim=-1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)                   # (B, H, T, hd)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)              # (B, H, T, hd)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    """Gated MLP:  out = (silu(x W_gate) * (x W_up)) W_down.   x: (B, T, D) -> hidden (B, T, F) -> (B, T, D)."""

    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Embed(nn.Module):
    """Unit 0: token ids (B, T) -> vectors (B, T, D)."""

    def __init__(self, cfg):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)

    def forward(self, idx):
        return self.tok(idx)


class Block(nn.Module):
    """Units 1..L: one pre-norm transformer layer.  x: (B, T, D) -> (B, T, D).

    The RoPE tables are buffers (not weights): they are never sharded, sent or updated.
    """

    def __init__(self, cfg, cos, sin):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), SwiGLU(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.cos, self.sin)
        return x + self.mlp(self.norm2(x))


class Head(nn.Module):
    """Unit L+1: final RMSNorm and the LM head.  x: (B, T, D) -> logits (B, T, V)."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x):
        return self.head(self.norm(x))


class GPT(nn.Module):
    """The decoder as an nn.ModuleList of units, run one after the other.

    Example:
        cfg = Config()                    # 5,310,720 weights
        model = GPT(cfg)
        idx = torch.tensor([[12, 7, 300]])  # (B=1, T=3) token ids
        model(idx).shape                    # torch.Size([1, 3, 4096])

    Initialisation (std of the normal each matrix is drawn from), as in Assignment 11:
        embedding 0.02;  qkv, gate, up 1/sqrt(fan_in);  attn.proj, mlp.down 1/sqrt(fan_in)/sqrt(2L);
        head 1/D (first logits are about 0, so the first loss is about ln V = 8.32);  RMSNorm gains 1.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cos, sin = rope_tables(cfg)
        self.units = nn.ModuleList([Embed(cfg)] + [Block(cfg, cos, sin) for _ in range(cfg.n_layer)] + [Head(cfg)])
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        depth_scale = 1.0 / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("tok.weight"):
                nn.init.normal_(p, std=0.02)
            elif name.endswith("head.weight"):
                nn.init.normal_(p, std=1.0 / self.cfg.d_model)
            elif p.dim() == 2:                                   # nn.Linear weight (fan_out, fan_in)
                std = 1.0 / math.sqrt(p.shape[1])
                if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                    std *= depth_scale
                nn.init.normal_(p, std=std)

    def forward(self, idx):
        """idx (B, T) -> logits (B, T, V)."""
        x = idx
        for unit in self.units:
            x = unit(x)
        return x


def count_params(cfg):
    """Hand formula for the number of weights.

        embedding V*D + L * (attention 4 D^2 + MLP 3 D F + two norms 2 D) + final norm D + head V*D
    Example: Config() -> 4096*256 + 4*(4*256^2 + 3*256*704 + 2*256) + 256 + 4096*256 = 5,310,720
    """
    V, D, F_, L = cfg.vocab_size, cfg.d_model, cfg.d_ff, cfg.n_layer
    return V * D + L * (4 * D * D + 3 * D * F_ + 2 * D) + D + V * D


def unit_sizes(cfg):
    """Weights per unit, in model order.  Example: Config() -> [1048576, 803328, 803328, 803328, 803328, 1048832]"""
    V, D, F_, L = cfg.vocab_size, cfg.d_model, cfg.d_ff, cfg.n_layer
    return [V * D] + [4 * D * D + 3 * D * F_ + 2 * D] * L + [D + V * D]
