"""
models/modules/losses.py
-------------------------
Objective-function components.

`BidirectionalTripletRankingLoss` and `dove_total_loss` are **faithful
transcriptions** of DOVE Eq (30) and Eq (31). `AFRNetObjective` is this
repository's composition, which keeps the **existing** primary objective
(BiomedCLIP's symmetric InfoNCE, `models/baseline.py::clip_loss`) and adds
OT-Align as an auxiliary term.

    DOVE Eq (30)   L(V,T) = sum_That [a - S(V,T) + S(V,That)]_+
                          + sum_Vhat [a - S(V,T) + S(Vhat,T)]_+
    DOVE Eq (31)   L_total = L(V_MR, T_RG) + lambda_g * L(V_M, T_G)

Paper hyperparameters (§IV-B): margin `alpha = 0.2`, `lambda_g = 10.0`.

**The primary objective is NOT silently changed.** `AFRNetObjective` defaults
to `primary="clip"`, reproducing the current baseline loss exactly, so that
baseline numbers stay comparable. Setting `primary="triplet"` switches to the
paper's Eq (30) and is an explicit, logged decision - see
docs/FIDELITY_AUDIT.md MOD-LOSS-1.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DOVE Eq (30)
# ---------------------------------------------------------------------------

class BidirectionalTripletRankingLoss(nn.Module):
    """DOVE Eq (30), the bidirectional triplet ranking loss.

    For a mini-batch of matched pairs `(V_i, T_i)`, every other sample in the
    batch is a negative. With `S` the cosine similarity and `[x]_+ = max(x,0)`:

        L = sum_i sum_{j != i} [alpha - S(V_i,T_i) + S(V_i,T_j)]_+     (i2t)
          + sum_i sum_{j != i} [alpha - S(V_i,T_i) + S(V_j,T_i)]_+     (t2i)

    Args:
        margin: `alpha`. Paper §IV-B uses 0.2.
        max_violation: if True, use only the hardest negative per anchor
            (VSE++ style) instead of summing over all negatives. The paper
            writes a **sum** (Eq 30), so the default is False. Switching this
            on is an algorithmic modification.
        reduction: "sum" follows Eq (30) literally. "mean" divides by the
            batch size, which only rescales the gradient but makes the value
            comparable across batch sizes; it is the default here because
            `lambda_g = 10.0` was tuned at batch size 100 and a raw sum is
            batch-size dependent. Recorded as MOD-LOSS-2.
    """

    def __init__(
        self,
        margin: float = 0.2,
        max_violation: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}")
        self.margin = float(margin)
        self.max_violation = max_violation
        self.reduction = reduction

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """`v`, `t`: [B, D]. Returns a scalar."""
        if v.shape != t.shape:
            raise ValueError(f"v {tuple(v.shape)} and t {tuple(t.shape)} must match")
        b = v.shape[0]
        if b < 2:
            # No negatives exist, so Eq (30) is an empty sum.
            return v.new_zeros(())

        # S(.,.) is cosine (paper: "S(.,.) is the Cosine function").
        sim = F.normalize(v, dim=-1) @ F.normalize(t, dim=-1).transpose(0, 1)  # [B, B]
        pos = sim.diagonal().view(b, 1)                                        # S(V_i,T_i)

        # i2t: vary the text, anchor the image -> compare along rows.
        cost_i2t = (self.margin - pos + sim).clamp_min(0)
        # t2i: vary the image, anchor the text -> compare along columns.
        cost_t2i = (self.margin - pos.transpose(0, 1) + sim).clamp_min(0)

        # Exclude the positive pair itself (j == i).
        eye = torch.eye(b, dtype=torch.bool, device=v.device)
        cost_i2t = cost_i2t.masked_fill(eye, 0)
        cost_t2i = cost_t2i.masked_fill(eye, 0)

        if self.max_violation:
            total = cost_i2t.max(dim=1).values.sum() + cost_t2i.max(dim=0).values.sum()
        else:
            total = cost_i2t.sum() + cost_t2i.sum()

        return total / b if self.reduction == "mean" else total


def dove_total_loss(
    v_mr: torch.Tensor,
    t_rg: torch.Tensor,
    v_m: torch.Tensor,
    t_g: torch.Tensor,
    ranking: BidirectionalTripletRankingLoss,
    lambda_g: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """DOVE Eq (31): `L(V_MR,T_RG) + lambda_g * L(V_M,T_G)`.

    `L(V_M, T_G)` is the **global visual-semantic constraint**: an external
    constraint on the *pre-ROAM* representations, which the paper adds so that
    "the original visual and textual semantics remain unchanged".
    """
    final = ranking(v_mr, t_rg)
    glob = ranking(v_m, t_g)
    return {
        "loss": final + lambda_g * glob,
        "ranking_final": final.detach(),
        "global_constraint": glob.detach(),
    }


# ---------------------------------------------------------------------------
# AFR-Net composed objective
# ---------------------------------------------------------------------------

class ObjectiveOutput(NamedTuple):
    total: torch.Tensor
    parts: Dict[str, torch.Tensor]


class AFRNetObjective(nn.Module):
    """Composed training objective for AFR-Net.

        L_total = L_primary
                + lambda_ot * L_OT
                + lambda_g  * L_global          (optional, paper-inspired)

    `L_primary` is BiomedCLIP's symmetric InfoNCE by default, which is what
    `trainers/baseline_trainer.py` already optimises. Each extra term has its
    own coefficient and each is reported separately so a run's objective is
    never ambiguous.

    Args:
        primary: "clip" (default, preserves the existing baseline objective)
                 or "triplet" (DOVE Eq 30).
        lambda_ot: weight of the OT-Align auxiliary loss. 0 disables it.
        lambda_g:  weight of the Eq (31) global constraint. 0 disables it.
                   The paper's 10.0 was tuned for its own architecture and
                   loss scale; do not transplant it without a sweep.
        margin / max_violation: passed to the triplet loss (used by the
                   "triplet" primary and by the global constraint).
    """

    def __init__(
        self,
        primary: str = "clip",
        lambda_ot: float = 0.0,
        lambda_g: float = 0.0,
        margin: float = 0.2,
        max_violation: bool = False,
        triplet_reduction: str = "mean",
    ):
        super().__init__()
        if primary not in ("clip", "triplet"):
            raise ValueError(f"primary must be clip/triplet, got {primary!r}")
        self.primary = primary
        self.lambda_ot = float(lambda_ot)
        self.lambda_g = float(lambda_g)
        self.ranking = BidirectionalTripletRankingLoss(
            margin=margin, max_violation=max_violation, reduction=triplet_reduction
        )

    def forward(
        self,
        v_final: torch.Tensor,
        t_final: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        ot_loss: Optional[torch.Tensor] = None,
        v_global: Optional[torch.Tensor] = None,
        t_global: Optional[torch.Tensor] = None,
    ) -> ObjectiveOutput:
        parts: Dict[str, torch.Tensor] = {}

        if self.primary == "clip":
            if logit_scale is None:
                raise ValueError("primary='clip' requires logit_scale")
            primary = clip_contrastive_loss(v_final, t_final, logit_scale)
        else:
            primary = self.ranking(v_final, t_final)
        parts["primary"] = primary
        total = primary

        if self.lambda_ot != 0.0:
            if ot_loss is None:
                raise ValueError("lambda_ot != 0 but no ot_loss was supplied")
            parts["ot"] = ot_loss
            total = total + self.lambda_ot * ot_loss

        if self.lambda_g != 0.0:
            if v_global is None or t_global is None:
                raise ValueError("lambda_g != 0 but v_global/t_global were not supplied")
            glob = self.ranking(v_global, t_global)
            parts["global_constraint"] = glob
            total = total + self.lambda_g * glob

        parts["total"] = total
        return ObjectiveOutput(total=total, parts=parts)


def clip_contrastive_loss(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE, identical in form to `models/baseline.py::clip_loss`.

    Reimplemented here (rather than imported) so that the objective module has
    no dependency on the backbone wrapper, and so the two can be compared.
    `tests/test_losses.py::test_clip_loss_matches_baseline_implementation`
    asserts they agree numerically.
    """
    scale = logit_scale.exp().clamp(max=100.0) if logit_scale.numel() == 1 else logit_scale
    logits = scale * F.normalize(img_feats, dim=-1) @ F.normalize(txt_feats, dim=-1).T
    labels = torch.arange(img_feats.shape[0], device=img_feats.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
