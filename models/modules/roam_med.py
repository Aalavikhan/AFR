"""
models/modules/roam_med.py
---------------------------
ROAM-Med — biomedical adaptation of DOVE's Regional-Oriented Attention Module
(§III-C, Equations 17-28).

The **mathematics of IFA and IGA is transcribed from the paper** and preserved
exactly. What changes is only where the two visual inputs come from, because
BiomedCLIP exposes neither a ResNet multiscale pyramid nor an RoI detector.

    DOVE input              ROAM-Med input                    classification
    ---------------------   -------------------------------   --------------
    F_M, MSV encoder,       SpatialPyramidPool over the 196   ADAPT-ROAM-1
    Nm = 4 ResNet stages    ViT patch tokens (Nm = 63 by
                            default)
    F_R, RoI encoder,       DDTP-selected top-98 patches      ADAPT-ROAM-2
    Nr = 36 detections      (text-conditioned "regions")

Both adaptations keep the tensor contract IFA/IGA require - `[B, Nm, D]` and
`[B, Nr, D]` - so Eq (17)-(28) run unmodified. See docs/FIDELITY_AUDIT.md.

Fidelity-critical details that are easy to get wrong
----------------------------------------------------
* **Eq (19) and (26) use `sigma` (sigmoid), NOT softmax.** `S_MR` and `S_RG`
  are element-wise gates in `(0,1)`, un-normalised over both axes, with **no
  `1/sqrt(d)` scaling**. Substituting softmax attention would silently replace
  the module. Asserted by `tests/test_roam_med.py`.
* **Eq (20)/(21) residuals are cross-wired**: `F_{R+M} = S_MR F'_R + F'_M`
  adds the *query* side back, not the attended side.
* **Eq (22) concatenates along the TOKEN axis** (`Nm + Nr` rows) and applies a
  linear head over the feature axis. Eq (28) uses a *non-linear* head.
* **IGA is batch-coupled**: Eq (23) pools to one vector per sample, so Eq (26)
  is `[B, B]` across the mini-batch and Eq (27) mixes samples. This is the
  paper's behaviour and the default here, but it makes a sample's embedding
  depend on its batch-mates - a real hazard for retrieval evaluation. See
  `iga_mode` below and WARN-ROAM-1 in the audit.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from ._common import ResidualMLP, as_bool_mask, masked_mean


class ROAMOutput(NamedTuple):
    f_mr: torch.Tensor          # [B, Nm+Nr, D]  fused visual tokens  (Eq 22)
    f_rg: torch.Tensor          # [B, D]         guided text embedding (Eq 28)
    v_mr: torch.Tensor          # [B, D]         mean of f_mr          (Eq 29)
    s_mr: torch.Tensor          # [B, Nm, Nr]    IFA score matrix      (Eq 19)
    s_rg: torch.Tensor          # [B, B] or [B]  IGA score matrix      (Eq 26)


# ---------------------------------------------------------------------------
# IFA — Intra-modal Fusion Attention, Eq (17)-(22)
# ---------------------------------------------------------------------------

class IntraModalFusionAttention(nn.Module):
    """DOVE Eq (17)-(22), plus this project's optional soft attention prior.

    Args:
        dim: feature size `d`.
        prior_mode: how the soft attention prior modulates `S_MR`.
            "multiply" (default) - `S_MR <- S_MR * prior`, broadcast over the
                `Nm` axis. Keeps Eq (19)'s gate semantics: both factors are in
                `(0, 1]`-ish and the product is still an element-wise gate.
            "add_logit"          - `S_MR <- sigma(logits + log(prior))`, i.e.
                the prior biases the pre-sigmoid score.
            "none"               - ignore any prior (literal Eq 19).
        prior_normalise: if True, a supplied prior is rescaled to mean 1 over
            its valid entries, so applying it does not change the expected
            magnitude of `S_MR`. Recommended.
    """

    def __init__(
        self,
        dim: int = 768,
        prior_mode: str = "multiply",
        prior_normalise: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if prior_mode not in ("multiply", "add_logit", "none"):
            raise ValueError(
                f"prior_mode must be multiply/add_logit/none, got {prior_mode!r}"
            )
        self.dim = dim
        self.prior_mode = prior_mode
        self.prior_normalise = prior_normalise

        self.w_m = nn.Linear(dim, dim, bias=bias)   # Eq (17)
        self.w_r = nn.Linear(dim, dim, bias=bias)   # Eq (18)
        self.w_l = nn.Linear(dim, dim, bias=bias)   # Eq (22) linear head

    # ------------------------------------------------------------------
    def _apply_prior(
        self,
        logits: torch.Tensor,                 # [B, Nm, Nr] pre-sigmoid
        prior: Optional[torch.Tensor],        # [B, Nr]
        region_mask: Optional[torch.Tensor],  # [B, Nr]
        prior_is_logits: bool = True,
    ) -> torch.Tensor:
        """Return `S_MR` in (0,1), Eq (19) with the optional prior folded in."""
        if prior is None or self.prior_mode == "none":
            s = torch.sigmoid(logits)
        elif self.prior_mode == "add_logit":
            p = self._prepare_prior(prior, region_mask, prior_is_logits).clamp_min(1e-6)
            s = torch.sigmoid(logits + p.log().unsqueeze(1))
        else:  # "multiply"
            p = self._prepare_prior(prior, region_mask, prior_is_logits)
            s = torch.sigmoid(logits) * p.unsqueeze(1)

        if region_mask is not None:
            # An invalid region must contribute nothing in Eq (20)/(21).
            s = s * region_mask.unsqueeze(1).to(s.dtype)
        return s

    def _prepare_prior(
        self,
        prior: torch.Tensor,
        region_mask: Optional[torch.Tensor],
        prior_is_logits: bool,
    ) -> torch.Tensor:
        """Non-negative [B, Nr] prior, optionally rescaled to mean 1.

        `prior_is_logits` must be stated by the caller, not guessed from the
        values. An earlier draft sniffed `prior.min() < 0` to decide whether to
        apply a sigmoid, which made the transform **data-dependent**: the same
        prior would be squashed or not depending on the batch it arrived in.
        DDTP scores are cosine/tau and therefore logits; an external
        segmentor prior is normally already a probability in [0, 1].
        """
        p = prior.to(torch.float32)
        if prior_is_logits:
            p = torch.sigmoid(p)
        else:
            p = p.clamp_min(0.0)
        if region_mask is not None:
            p = p * region_mask.to(p.dtype)
        if self.prior_normalise:
            denom = p.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if region_mask is not None:
                count = region_mask.sum(dim=1, keepdim=True).clamp_min(1).to(p.dtype)
            else:
                count = torch.full_like(denom, float(p.shape[1]))
            p = p * (count / denom)
        return p

    # ------------------------------------------------------------------
    def forward(
        self,
        f_m: torch.Tensor,                      # [B, Nm, D]
        f_r: torch.Tensor,                      # [B, Nr, D]
        region_prior: Optional[torch.Tensor] = None,   # [B, Nr]
        region_mask: Optional[torch.Tensor] = None,    # [B, Nr]
        prior_is_logits: bool = True,
    ):
        if f_m.shape[-1] != self.dim or f_r.shape[-1] != self.dim:
            raise ValueError(
                f"IFA expects feature dim {self.dim}; got {f_m.shape[-1]} and {f_r.shape[-1]}"
            )
        rmask = as_bool_mask(region_mask, f_r) if region_mask is not None else None

        # Eq (17), (18)
        f_m_p = self.w_m(f_m)                                  # [B, Nm, D]
        f_r_p = self.w_r(f_r)                                  # [B, Nr, D]

        # Eq (19): sigmoid of the raw matrix product. No softmax, no 1/sqrt(d).
        logits = f_m_p @ f_r_p.transpose(1, 2)                 # [B, Nm, Nr]
        s_mr = self._apply_prior(logits, region_prior, rmask, prior_is_logits)

        # Eq (20): attend over regions, residual from the multiscale side.
        f_r_plus_m = s_mr @ f_r_p + f_m_p                      # [B, Nm, D]
        # Eq (21): transpose attend over scales, residual from the region side.
        f_m_plus_r = s_mr.transpose(1, 2) @ f_m_p + f_r_p      # [B, Nr, D]

        if rmask is not None:
            f_m_plus_r = f_m_plus_r * rmask.unsqueeze(-1).to(f_m_plus_r.dtype)

        # Eq (22): concatenate along the TOKEN axis, then a linear head.
        fused = torch.cat([f_r_plus_m, f_m_plus_r], dim=1)     # [B, Nm+Nr, D]
        f_mr = self.w_l(fused)

        if rmask is not None:
            n_m = f_m.shape[1]
            keep = torch.cat(
                [torch.ones(f_m.shape[0], n_m, dtype=torch.bool, device=f_m.device), rmask],
                dim=1,
            )
            f_mr = f_mr * keep.unsqueeze(-1).to(f_mr.dtype)
        else:
            keep = None

        return f_mr, s_mr, keep


# ---------------------------------------------------------------------------
# IGA — Inter-modal Guidance Attention, Eq (23)-(28)
# ---------------------------------------------------------------------------

class InterModalGuidanceAttention(nn.Module):
    """DOVE Eq (23)-(28).

    Args:
        dim: feature size `d`.
        mode:
            "batch" (default, **faithful**) - Eq (26) is `sigma(F'_R F'_G^T)`
                over the mini-batch, shape `[B, B]`, and Eq (27) mixes every
                sample's text vector with every other's. This is exactly what
                the paper specifies ("`Bc` is the batch size of text input").
            "diagonal" - keep only the self term, i.e. a per-sample scalar
                `s_i = sigma(F'_R[i] . F'_G[i])` and
                `F_{R+G}[i] = s_i * F'_G[i] + F'_G[i]`. This removes the
                batch coupling and makes inference batch-independent. It is an
                **algorithmic modification** (MOD-ROAM-1), not the paper's
                behaviour, and must be reported as such in any results.
    """

    def __init__(
        self,
        dim: int = 768,
        mode: str = "batch",
        mlp_hidden_mult: float = 4.0,
        mlp_activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        if mode not in ("batch", "diagonal"):
            raise ValueError(f"mode must be batch/diagonal, got {mode!r}")
        self.dim = dim
        self.mode = mode
        self.w_r = nn.Linear(dim, dim, bias=bias)   # Eq (24)
        self.w_g = nn.Linear(dim, dim, bias=bias)   # Eq (25)
        self.head = ResidualMLP(                    # Eq (28) non-linear head
            dim, hidden_mult=mlp_hidden_mult, activation=mlp_activation
        )

    def forward(
        self,
        f_r: torch.Tensor,                           # [B, Nr, D]
        f_g: torch.Tensor,                           # [B, Nc, D]
        text_mask: Optional[torch.Tensor] = None,    # [B, Nc]
        region_mask: Optional[torch.Tensor] = None,  # [B, Nr]
    ):
        # Eq (23): mean-pool each modality to one vector per sample.
        # Masked so that padding tokens and invalid regions do not dilute the
        # mean; with masks of None this is the paper's plain (1/N) sum.
        e_r = masked_mean(f_r, region_mask, dim=1)   # [B, D]
        e_g = masked_mean(f_g, text_mask, dim=1)     # [B, D]

        # Eq (24), (25)
        f_r_p = self.w_r(e_r)                        # [B, D]
        f_g_p = self.w_g(e_g)                        # [B, D]

        if self.mode == "batch":
            # Eq (26): sigmoid of the matrix product ACROSS the mini-batch.
            s_rg = torch.sigmoid(f_r_p @ f_g_p.transpose(0, 1))   # [B, B]
            # Eq (27): residual on the text side.
            f_r_plus_g = s_rg @ f_g_p + f_g_p                     # [B, D]
        else:
            s_rg = torch.sigmoid((f_r_p * f_g_p).sum(dim=-1))     # [B]
            f_r_plus_g = s_rg.unsqueeze(-1) * f_g_p + f_g_p       # [B, D]

        # Eq (28)
        f_rg = self.head(f_r_plus_g)
        return f_rg, s_rg


# ---------------------------------------------------------------------------
# ROAM-Med
# ---------------------------------------------------------------------------

class ROAMMed(nn.Module):
    """IFA + IGA, with the soft attention prior wired into IFA's Eq (19).

    Shapes:
        f_m:          [B, Nm, D]  multiscale visual tokens
        f_r:          [B, Nr, D]  regional visual tokens (e.g. DDTP output)
        f_g:          [B, Nc, D]  word-level text features (e.g. GTGA output)
        region_prior: [B, Nr]     optional soft prior over regions
        region_mask:  [B, Nr]     optional validity of each region slot
        text_mask:    [B, Nc]     optional validity of each text token
    """

    def __init__(
        self,
        dim: int = 768,
        prior_mode: str = "multiply",
        prior_normalise: bool = True,
        iga_mode: str = "batch",
        mlp_hidden_mult: float = 4.0,
        mlp_activation: str = "gelu",
    ):
        super().__init__()
        self.dim = dim
        self.ifa = IntraModalFusionAttention(
            dim, prior_mode=prior_mode, prior_normalise=prior_normalise
        )
        self.iga = InterModalGuidanceAttention(
            dim, mode=iga_mode, mlp_hidden_mult=mlp_hidden_mult,
            mlp_activation=mlp_activation,
        )

    def forward(
        self,
        f_m: torch.Tensor,
        f_r: torch.Tensor,
        f_g: torch.Tensor,
        region_prior: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        prior_is_logits: bool = True,
    ) -> ROAMOutput:
        f_mr, s_mr, fused_mask = self.ifa(
            f_m, f_r, region_prior=region_prior, region_mask=region_mask,
            prior_is_logits=prior_is_logits,
        )
        f_rg, s_rg = self.iga(
            f_r, f_g, text_mask=text_mask, region_mask=region_mask
        )
        # Eq (29): V_MR is the mean of the fused visual tokens.
        v_mr = masked_mean(f_mr, fused_mask, dim=1)
        return ROAMOutput(f_mr=f_mr, f_rg=f_rg, v_mr=v_mr, s_mr=s_mr, s_rg=s_rg)
