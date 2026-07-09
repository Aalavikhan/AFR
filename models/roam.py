"""
models/roam.py
--------------
Module 1 — ROAM-Med (Region-Oriented Alignment Module, medical adaptation).

Ports DOVE's IFA/IGA region-guidance to a ViT + PubMedBERT backbone.
Fires ONCE per forward pass, right after get_patch_tokens / get_word_tokens,
before DDTP and OT-Align.

Key ROCO adaptation
-------------------
ROCO is heterogeneous radiology (X-ray, CT, MRI, US, angio...), so there is
no single segmentation model that gives a valid `P_seg` across all of it.
Instead the region prior is derived FROM THE BACKBONE ITSELF:
    P_seg = min-max normalised cosine( patch_i , CLS )
This is the "soft prior" version of the plan's attention-fallback (§5.5 step 5),
and it needs no external segmentor and no precompute cache.

If you later run a single-modality subset and DO have a real segmentation prior,
pass it in as `P_seg` and it overrides the backbone-derived one.

Shapes (ViT-B/16 + PubMedBERT_256):
    V  patch tokens : [B, 196, 768]
    W  word tokens  : [B, 256, 768]
    -> V_MR         : [B, 196, 768]
    -> T_RG         : [B, 256, 768]
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROAMMed(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_scales: int = 1,        # 1 = final-layer only (ROCO v1). >1 enables α-fusion.
        num_region_protos: int = 4, # learned region prototypes for the IGA branch
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        # STEP 1 — multi-scale fusion weights (softmaxed). Only used if num_scales > 1.
        if num_scales > 1:
            self.scale_logits = nn.Parameter(torch.zeros(num_scales))

        # STEP 3 — IFA (visual): query = fused patches, key/value = region features
        self.ifa_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ifa_norm = nn.LayerNorm(dim)

        # STEP 4 — IGA (text): learned region prototypes avoid the degenerate
        # length-1 key/value the plan warns about (§5.8).
        self.region_protos = nn.Parameter(torch.randn(num_region_protos, dim) * 0.02)
        self.iga_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.iga_norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    @staticmethod
    def _derive_prior(V: torch.Tensor, v_cls: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Backbone-derived soft region prior in [0, 1], shape [B, N].
        Uses cosine(patch, reference) where reference is the CLS token if given,
        else the mean patch. Min-max normalised per sample.
        """
        ref = v_cls if v_cls is not None else V.mean(dim=1)          # [B, D]
        sim = F.cosine_similarity(V, ref.unsqueeze(1), dim=-1)       # [B, N]
        lo = sim.amin(dim=1, keepdim=True)
        hi = sim.amax(dim=1, keepdim=True)
        return (sim - lo) / (hi - lo + 1e-6)                         # [B, N]

    # ------------------------------------------------------------------
    def visual_branch(
        self,
        V: torch.Tensor,                                  # [B, N, D]
        P_seg: Optional[torch.Tensor] = None,
        V_multi: Optional[Sequence[torch.Tensor]] = None,
        v_cls: Optional[torch.Tensor] = None,
    ):
        """IFA. Text-free -> usable for decoupled image encoding.
        Returns V_MR [B,N,D] and the region descriptor r_bar [B,D]."""
        # STEP 1 — multi-scale fusion (identity when single scale)
        if self.num_scales > 1 and V_multi is not None:
            alpha = torch.softmax(self.scale_logits, dim=0)
            V_fused = sum(a * v for a, v in zip(alpha, V_multi))
        else:
            V_fused = V

        # STEP 2 — soft region prior (backbone-derived unless supplied)
        if P_seg is None:
            P_flat = self._derive_prior(V_fused, v_cls)
        else:
            P_flat = P_seg.flatten(1) if P_seg.dim() == 3 else P_seg
        F_R = P_flat.unsqueeze(-1) * V_fused                         # [B, N, D]

        # STEP 3 — IFA
        ifa_out, _ = self.ifa_attn(V_fused, F_R, F_R, need_weights=False)
        V_MR = self.ifa_norm(ifa_out + V_fused)                      # [B, N, D]
        r_bar = F_R.mean(dim=1)                                      # [B, D]
        return V_MR, P_flat, r_bar

    def text_branch(
        self,
        W: torch.Tensor,                                  # [B, L, D]
        r_bar: Optional[torch.Tensor] = None,             # [B, D] or None (neutral)
    ):
        """IGA. If r_bar is None, uses learned region prototypes only ->
        image-free, usable for decoupled text encoding."""
        B = W.shape[0]
        if r_bar is None:
            kv = self.region_protos.unsqueeze(0).expand(B, -1, -1)   # [B, k, D]
        else:
            kv = self.region_protos.unsqueeze(0) + r_bar.unsqueeze(1)
        iga_out, _ = self.iga_attn(W, kv, kv, need_weights=False)
        T_RG = self.iga_norm(iga_out + W)                            # [B, L, D]
        return T_RG

    def forward(self, V, W, P_seg=None, V_multi=None, v_cls=None):
        """Coupled path (training convenience)."""
        V_MR, P_flat, r_bar = self.visual_branch(V, P_seg, V_multi, v_cls)
        T_RG = self.text_branch(W, r_bar)
        return V_MR, T_RG, P_flat
