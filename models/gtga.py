"""
models/gtga.py
--------------
Module 2 — GTGA (Gated Text-Guidance / phrase-aware text enhancement).

Replaces DOVE's DTGA bidirectional-GRU (redundant on top of an already
bidirectional PubMedBERT) with span-aware depthwise-separable 1D convolutions:
    kernel 3 -> trigram spans   ("spiculated nodule")
    kernel 5 -> pentagram spans ("right lower lobe consolidation")
followed by a gated self-attention refinement.

Runs on T_RG (region-activated words from ROAM). Outputs F_G, same shape,
with phrase-level structure baked in. F_G flows into OT-Align and is pooled
for the text global.

Shapes:
    T_RG : [B, 256, 768]  ->  F_G : [B, 256, 768]
"""

from typing import Optional

import torch
import torch.nn as nn


class GTGA(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim

        # Depthwise 1D convs over the sequence axis (groups=dim => per-channel).
        self.dw3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # Pointwise mixing after depthwise (depthwise-separable completion).
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)

        self.span_norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        T_RG: torch.Tensor,                              # [B, L, D]
        key_padding_mask: Optional[torch.Tensor] = None, # [B, L] True = PAD
    ) -> torch.Tensor:
        x = T_RG.transpose(1, 2)                         # [B, D, L]
        span = self.dw3(x) + self.dw5(x)                 # [B, D, L]
        span = self.pw(span).transpose(1, 2)             # [B, L, D]
        span = self.span_norm(span + T_RG)               # residual + norm

        g = torch.sigmoid(self.gate(span))               # [B, L, D]
        gated = g * span                                 # phrase-gated tokens

        attn_out, _ = self.attn(
            gated, gated, gated,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        F_G = self.attn_norm(attn_out + gated)           # [B, L, D]
        return F_G
