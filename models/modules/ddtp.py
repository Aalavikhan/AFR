"""
models/modules/ddtp.py
-----------------------
DDTP — Diagnostic-Driven Token Pruning (this project's **custom** module).

DDTP has **no counterpart in DOVE**. The paper obtains its `Nr = 36` regional
features from a pretrained RoI detector (§III-A-1); it never scores or selects
patches. DDTP replaces that detector with text-conditioned selection over the
ViT patch grid, which is what makes a frozen-backbone, detector-free regional
representation possible at all.

Design (all of it this repository's choice — see docs/MODULE_SPEC.md §4):

    1. project patches and the text query into a shared scoring space
    2. L2-normalise both, so the score is a cosine similarity in [-1, 1]
    3. divide by a temperature `tau`
    4. mask out invalid patches with -inf
    5. take the top `k` scores, `k = round(ratio * Np)` by default
    6. sort the chosen indices ASCENDING so spatial order is preserved
    7. gather features at those indices, and weight them by a differentiable
       function of their own scores so the scorer receives gradient

Gradient behaviour (important, and tested)
------------------------------------------
`torch.topk` returns **indices**, which are discrete and therefore
non-differentiable: no gradient flows through *which* patches were chosen.
Gradient reaches the scorer only through the **score weighting** of step 7.
With `score_weighting="none"` the scorer would receive no gradient at all and
would never train, so that mode exists only for ablation. The default
`"softmax_norm"` multiplies each selected feature by `k * softmax(s_sel)`,
which sums to `k` and so leaves the feature scale unchanged in expectation
while still being differentiable.

This is the standard limitation of hard top-k selection. If end-to-end
differentiable selection is ever required, a perturbed-top-k or Gumbel
relaxation would be needed; that would be an algorithmic change and is listed
as an open item in docs/FIDELITY_AUDIT.md.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import as_bool_mask


class DDTPOutput(NamedTuple):
    """Everything downstream modules and tests need from a selection."""

    features: torch.Tensor      # [B, k, D] selected patch features (weighted)
    indices: torch.Tensor       # [B, k] int64, ASCENDING original patch indices
    scores: torch.Tensor        # [B, k] relevance score of each selected patch
    mask: torch.Tensor          # [B, k] bool, True where the slot is a real patch
    all_scores: torch.Tensor    # [B, Np] score of every patch, before selection


class DDTP(nn.Module):
    """Text-conditioned patch selection.

    Args:
        dim:        patch/text feature size (768).
        score_dim:  width of the shared scoring space. `None` -> `dim`.
        ratio:      retention ratio. 0.5 over 196 patches selects **98**.
        num_keep:   explicit `k`, overriding `ratio` when not None.
        tau:        temperature dividing the cosine score.
        score_weighting:
                    "softmax_norm" (default) -> features * k*softmax(s_sel)
                    "sigmoid"                -> features * sigmoid(s_sel)
                    "none"                   -> no weighting; scorer gets NO
                                                gradient (ablation only)
        shared_projection:
                    True  -> one projection for both modalities
                    False -> separate projections (default)

    Shapes:
        patch_tokens: [B, Np, dim]   spatial patches only, CLS already removed
        text_query:   [B, dim]       e.g. masked-mean of GTGA output
        patch_mask:   [B, Np] bool   optional; True = eligible patch
    """

    def __init__(
        self,
        dim: int = 768,
        score_dim: Optional[int] = None,
        ratio: float = 0.5,
        num_keep: Optional[int] = None,
        tau: float = 0.07,
        score_weighting: str = "softmax_norm",
        shared_projection: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        if num_keep is not None and num_keep < 1:
            raise ValueError(f"num_keep must be >= 1, got {num_keep}")
        if score_weighting not in ("softmax_norm", "sigmoid", "none"):
            raise ValueError(
                f"score_weighting must be softmax_norm/sigmoid/none, got {score_weighting!r}"
            )
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")

        self.dim = dim
        self.ratio = float(ratio)
        self.num_keep = num_keep
        self.tau = float(tau)
        self.score_weighting = score_weighting

        sdim = score_dim or dim
        self.patch_proj = nn.Linear(dim, sdim, bias=bias)
        self.text_proj = self.patch_proj if shared_projection else nn.Linear(dim, sdim, bias=bias)

    # ------------------------------------------------------------------
    def resolve_k(self, num_patches: int) -> int:
        """How many patches this configuration keeps out of `num_patches`.

        `ratio=0.5`, `num_patches=196` -> 98 (the user's intended setting).
        Always clamped to `[1, num_patches]`.
        """
        if self.num_keep is not None:
            return int(min(self.num_keep, num_patches))
        k = int(round(self.ratio * num_patches))
        return max(1, min(k, num_patches))

    # ------------------------------------------------------------------
    def score(
        self,
        patch_tokens: torch.Tensor,
        text_query: torch.Tensor,
    ) -> torch.Tensor:
        """Relevance of every patch to the text, [B, Np].

        Cosine similarity in the projected space, divided by `tau`.
        """
        p = F.normalize(self.patch_proj(patch_tokens), dim=-1)   # [B, Np, S]
        t = F.normalize(self.text_proj(text_query), dim=-1)      # [B, S]
        return torch.einsum("bnd,bd->bn", p, t) / self.tau

    # ------------------------------------------------------------------
    def forward(
        self,
        patch_tokens: torch.Tensor,
        text_query: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> DDTPOutput:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be [B, Np, D], got {tuple(patch_tokens.shape)}")
        if patch_tokens.shape[-1] != self.dim:
            raise ValueError(f"expected patch dim {self.dim}, got {patch_tokens.shape[-1]}")
        if text_query.shape != (patch_tokens.shape[0], self.dim):
            raise ValueError(
                f"text_query must be [B, {self.dim}], got {tuple(text_query.shape)}"
            )

        b, n_p, _ = patch_tokens.shape
        k = self.resolve_k(n_p)

        all_scores = self.score(patch_tokens, text_query)        # [B, Np]

        # Invalid patches must never be selected. -inf also keeps them out of
        # the softmax weighting below.
        if patch_mask is not None:
            valid = as_bool_mask(patch_mask, patch_tokens)
            masked_scores = all_scores.masked_fill(~valid, float("-inf"))
        else:
            valid = None
            masked_scores = all_scores

        # `torch.topk` breaks ties by lower index on CPU, but that is not
        # guaranteed on CUDA. Sorting the selected indices ascending makes the
        # RESULT deterministic in order regardless of tie-break order; which
        # member of an exact tie is picked can still vary by device, and that
        # is documented rather than hidden.
        topk = masked_scores.topk(k, dim=1)
        indices, _ = topk.indices.sort(dim=1)                    # [B, k] ascending

        sel_scores = all_scores.gather(1, indices)               # [B, k]

        # A slot is real only if its source patch was valid. With fewer than
        # `k` valid patches, -inf entries can be selected; mark them invalid
        # instead of silently passing padding downstream as a "region".
        if valid is not None:
            sel_mask = valid.gather(1, indices)
        else:
            sel_mask = torch.ones(b, k, dtype=torch.bool, device=patch_tokens.device)

        features = patch_tokens.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, self.dim)
        )                                                        # [B, k, D]

        # Differentiable path to the scorer.
        if self.score_weighting != "none":
            scores_for_w = sel_scores.masked_fill(~sel_mask, float("-inf"))
            if self.score_weighting == "softmax_norm":
                w = torch.softmax(scores_for_w, dim=1) * k
                w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            else:  # "sigmoid"
                w = torch.sigmoid(sel_scores) * sel_mask.to(sel_scores.dtype)
            features = features * w.unsqueeze(-1).to(features.dtype)

        features = features * sel_mask.unsqueeze(-1).to(features.dtype)

        return DDTPOutput(
            features=features,
            indices=indices,
            scores=sel_scores,
            mask=sel_mask,
            all_scores=all_scores,
        )
