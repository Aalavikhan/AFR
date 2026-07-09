"""
models/ddtp.py
--------------
Module 3 — DDTP (Discriminative Dynamic Token Pruning).

Ports DFIM's DVE: a learnable multimodal prompt scores every patch by
text-conditional relevance, then the top-M patches are retained and the rest
dropped before OT-Align sees them. Removes background/artifact patches that
would otherwise add noise to the transport plan.

ROCO v1 simplification (flagged): we use FIXED top-M in BOTH train and eval so
retrieval batches cleanly. The plan's dynamic-Sparsemax-at-inference is left as
an ablation TODO (variable M breaks batched Sinkhorn). Sparsemax weights are
still computed and exposed so the ablation is one flag away.

Gradient note: DFS is computed over ALL 196 patches, so gradient flows back
through every patch's score; only the *reconstruction* of retained patches is
truncated by top-M selection.

Shapes:
    V_MR : [B, 196, 768]   w_eos : [B, 768]   v_cls : [B, 768]
    -> v_final : [B, 768]   V_tilde : [B, M, 768]   mask : [B, 196] bool
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from entmax import sparsemax
except Exception:  # entmax optional at import time
    sparsemax = None


class DDTP(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_prompts: int = 8,
        num_heads: int = 8,
        keep_ratio: float = 0.5,
        min_keep: int = 20,
        num_patches: int = 196,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.M = max(min_keep, math.ceil(num_patches * keep_ratio))  # 98 for 196 @ 0.5

        self.prompts = nn.Parameter(torch.randn(num_prompts, dim) * 0.02)
        self.prompt_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.agg_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.lam = nn.Parameter(torch.tensor(0.5))  # residual fusion weight

    @staticmethod
    def _minmax(x: torch.Tensor) -> torch.Tensor:
        lo = x.amin(dim=1, keepdim=True)
        hi = x.amax(dim=1, keepdim=True)
        return (x - lo) / (hi - lo + 1e-6)

    def forward(
        self,
        V_MR: torch.Tensor,     # [B, N, D]
        w_eos: torch.Tensor,    # [B, D]
        v_cls: torch.Tensor,    # [B, D]
    ):
        B, N, D = V_MR.shape

        # STEP 1 — multimodal prompt attention over patches ⊕ text summary
        Q = self.prompt_norm(self.prompts).unsqueeze(0).expand(B, -1, -1)   # [B, P, D]
        kv = torch.cat([V_MR, w_eos.unsqueeze(1)], dim=1)                    # [B, N+1, D]
        prompt_out, _ = self.cross_attn(Q, kv, kv, need_weights=False)       # [B, P, D]
        q = prompt_out.mean(dim=1)                                           # [B, D]

        # STEP 2 — per-patch relevance A_w
        scores = (V_MR @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(D)         # [B, N]
        A_w = torch.softmax(scores, dim=1)                                   # [B, N]

        # STEP 3 — Discriminant Feature Score (both terms min-max'd to [0,1])
        cls_sim = F.cosine_similarity(V_MR, v_cls.unsqueeze(1), dim=-1)      # [B, N]
        DFS = 0.5 * (self._minmax(cls_sim) + self._minmax(A_w))             # [B, N]

        # STEP 4 — fixed top-M retention (stable batching)
        top_idx = DFS.topk(self.M, dim=1).indices                           # [B, M]
        mask = torch.zeros(B, N, dtype=torch.bool, device=V_MR.device)
        mask.scatter_(1, top_idx, True)

        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)                   # [B, M, D]
        retained_raw = torch.gather(V_MR, 1, idx_exp)                       # [B, M, D]

        # Gate retained patches by their score. This is what keeps the scoring
        # network (prompts + cross-attn -> A_w -> DFS) in the differentiable
        # graph. WITHOUT this, topk indices are detached and the scorer trains
        # dead — the exact silent-death failure mode from MISDD-MM.
        ret_score = torch.gather(DFS, 1, top_idx).unsqueeze(-1)            # [B, M, 1] in [0,1]
        retained = retained_raw * (0.5 + ret_score)                        # keep magnitude ~O(1)

        # sparsemax weights exposed for the inference-time ablation (unused in v1)
        sparse_w = sparsemax(scores, dim=1) if sparsemax is not None else None

        # STEP 5 — aggregate retained patches to a refined global feature
        pooled = retained.max(dim=1).values                                 # [B, D]
        v_hat = self.agg_mlp(pooled)                                        # [B, D]
        v_final = self.lam * v_hat + v_cls                                  # [B, D]

        # STEP 6 — normalise retained for OT
        V_tilde = F.normalize(retained, dim=-1)                             # [B, M, D]

        return v_final, V_tilde, mask, {"DFS": DFS, "A_w": A_w, "sparse_w": sparse_w}
