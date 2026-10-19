"""
models/modules/gated_attention.py
----------------------------------
Gated Self-Attention `G(·)` — DOVE §III-B-1, Equations (6)-(11).

This is a **faithful transcription** of the paper's operations, kept as a
standalone, separately testable component. It is the one place in this
repository that reproduces a DOVE attention mechanism verbatim.

    (6)   X_tilde = G(X)
    (7)   Q = X W_Q + b_Q ;  K = X W_K + b_K ;  V = X W_V + b_V
    (8)   G = sigma((Q (*) K) W_A + b_A)
    (9)   Q' = G (*) Q
    (10)  K' = G (*) K
    (11)  X_tilde = Softmax(Q' K'^T / sqrt(d)) V

where `(*)` is the element-wise product.

Fidelity notes (see docs/FIDELITY_AUDIT.md, FAITH-GA-*):

* Eq (8) is **not** a conventional gate. It takes the *element-wise* product of
  Q and K, pushes it through **one** fully-connected layer, then a sigmoid.
* Only Q and K are gated. `V` is deliberately left ungated (Eq 9-10).
* §IV-B states "there are 2 headers for text feature encoding", so the paper's
  own implementation is 2-head even though Eq (11) is written single-head.
  `num_heads` defaults to 2 accordingly. The `1/sqrt(d)` of Eq (11) is applied
  with `d` = **per-head** dimension, which is the standard reading of
  multi-head scaled dot-product attention.
* GTGA does **not** use this module by default. The user's GTGA is a depthwise
  convolutional design; gated self-attention is available there only behind an
  explicit opt-in flag.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import as_bool_mask


class GatedSelfAttention(nn.Module):
    """DOVE Eq (6)-(11).

    Args:
        dim:        feature size `d`.
        num_heads:  attention heads. Paper §IV-B uses 2.
        gate_on:    which tensors the Eq (8) gate multiplies. "qk" is the
                    paper's behaviour (Eq 9-10). Provided only so that an
                    ablation can be run; any value other than "qk" is an
                    algorithmic modification.
        bias:       include the `b_Q, b_K, b_V, b_A` biases of Eq (7)/(8).

    Shapes:
        input  x: [B, N, d]
        mask:     [B, N] bool, True = valid (optional)
        output:   [B, N, d]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        gate_on: str = "qk",
        bias: bool = True,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        if gate_on not in ("qk", "q", "k", "none"):
            raise ValueError(f"gate_on must be one of qk/q/k/none, got {gate_on!r}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gate_on = gate_on

        # Eq (7)
        self.w_q = nn.Linear(dim, dim, bias=bias)
        self.w_k = nn.Linear(dim, dim, bias=bias)
        self.w_v = nn.Linear(dim, dim, bias=bias)
        # Eq (8): a single FC layer over the element-wise product Q (*) K
        self.w_a = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = attn_dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        if d != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {d}")

        # --- Eq (7) ----------------------------------------------------------
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # --- Eq (8): gate from the element-wise product of Q and K -----------
        gate = torch.sigmoid(self.w_a(q * k))

        # --- Eq (9) and (10): gate Q and K, leave V untouched ----------------
        if self.gate_on in ("qk", "q"):
            q = gate * q
        if self.gate_on in ("qk", "k"):
            k = gate * k

        # --- Eq (11): multi-head scaled dot-product attention ----------------
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        qh, kh, vh = split(q), split(k), split(v)
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            valid = as_bool_mask(mask, x)  # [B, N]
            # Mask keys: a padded token must never receive attention mass.
            scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
            # A row whose every key is masked would softmax to NaN. That can
            # only happen for an all-padding sample; neutralise those rows.
            dead = (~valid).all(dim=1)  # [B]
            if dead.any():
                scores = scores.masked_fill(dead[:, None, None, None], 0.0)

        attn = scores.softmax(dim=-1)
        if self.attn_dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_dropout)

        out = (attn @ vh).transpose(1, 2).reshape(b, n, d)

        if mask is not None:
            out = out * as_bool_mask(mask, x).unsqueeze(-1).to(out.dtype)
        return out
