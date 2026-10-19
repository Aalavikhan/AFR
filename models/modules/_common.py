"""
models/modules/_common.py
-------------------------
Shared helpers for the AFR-Net research modules.

Nothing in this file is paper-specific. It exists so that masking, pooling and
multiscale derivation are implemented **once** and tested once, rather than
re-derived (and subtly diverged) inside each module.

Masking convention used everywhere in `models/modules/`:
    mask[b, i] == True  -> position i of sample b is VALID
    mask[b, i] == False -> position i is padding / invalid
A mask of `None` always means "everything is valid".
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def as_bool_mask(mask: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """Return a [B, N] bool mask, materialising an all-valid mask when None.

    `like` is the [B, N, D] tensor the mask applies to.
    """
    if mask is None:
        return torch.ones(like.shape[0], like.shape[1], dtype=torch.bool, device=like.device)
    if mask.dtype != torch.bool:
        mask = mask != 0
    if mask.shape != like.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match tokens {tuple(like.shape[:2])}"
        )
    return mask.to(like.device)


def zero_invalid(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero out padded positions of [B, N, D] so they cannot leak downstream.

    Called before and after every convolution in GTGA: a depthwise kernel of
    width k mixes neighbours, so a non-zero padded position would contaminate
    up to k//2 valid positions on each side.
    """
    if mask is None:
        return x
    return x * as_bool_mask(mask, x).unsqueeze(-1).to(x.dtype)


def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean of [B, N, D] over `dim`, counting only valid positions.

    This is the masked form of DOVE Eq (23)/(29). With `mask=None` it reduces
    exactly to the paper's unweighted `(1/N) Σ_i x_i`.

    A sample with zero valid positions yields a zero vector rather than NaN.
    """
    if mask is None:
        return x.mean(dim=dim)
    m = as_bool_mask(mask, x).unsqueeze(-1).to(x.dtype)
    total = (x * m).sum(dim=dim)
    count = m.sum(dim=dim).clamp_min(eps)
    return total / count


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """`MLP(x) + x` — the residual MLP of DOVE Eq (1), (16) and (28).

    The paper does not state the hidden width or activation of these MLPs, so
    both are explicit constructor arguments. Defaults (4x expansion, GELU) are
    this repository's choice and are recorded as such in FIDELITY_AUDIT.md.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 4.0,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = max(1, int(round(dim * hidden_mult)))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = _activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x)))) + x


class TwoLayerFC(nn.Module):
    """`F(·)` of DOVE Eq (14): "2-layer fully connected layer".

    The paper does not state the output activation. Because Eq (15) uses the
    result as a multiplicative "probability matrix", a bounded non-negative
    range is implied; `out_activation="sigmoid"` is this repository's default
    and is labelled an INFERENCE in FIDELITY_AUDIT.md. Set
    `out_activation="identity"` to follow the literal text of Eq (14).
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 1.0,
        activation: str = "relu",
        out_activation: str = "sigmoid",
    ):
        super().__init__()
        hidden = max(1, int(round(dim * hidden_mult)))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = _activation(activation)
        self.fc2 = nn.Linear(hidden, dim)
        self.out_act = _activation(out_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.fc2(self.act(self.fc1(x))))


def _activation(name: str) -> nn.Module:
    table = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"unsupported activation {name!r}; choose from {sorted(table)}")
    return table[key]()


# ---------------------------------------------------------------------------
# Multiscale derivation (biomedical adaptation of DOVE's MSV encoder)
# ---------------------------------------------------------------------------

class SpatialPyramidPool(nn.Module):
    """Derive a multiscale token set from a flat ViT patch grid.

    **Why this exists.** DOVE's `F_M` (Eq 1) comes from a ResNet-50 MSV encoder
    that emits one token per stage (`Nm = 4`, "the semantic-level features from
    layer1..layer4"). A ViT has no spatial downsampling, so that exact input
    cannot be obtained from BiomedCLIP's patch tokens. This module builds the
    least invasive substitute: adaptive average pooling of the `14x14` patch
    grid at several scales, concatenated into `Nm` tokens.

    This is an **input adaptation**, not a change to IFA's algorithm — IFA
    (Eq 17-22) still receives an `[B, Nm, D]` multiscale tensor and an
    `[B, Nr, D]` regional tensor and computes exactly the paper's operations.
    The adaptation is recorded in FIDELITY_AUDIT.md as ADAPT-ROAM-1.

    `scales=(1, 2, 3, 7)` gives `Nm = 1 + 4 + 9 + 49 = 63` tokens.
    `scales=(1, 1, 1, 1)` would give the paper's `Nm = 4` but is degenerate
    (four copies of the global mean), so it is not the default.
    """

    def __init__(self, scales: Sequence[int] = (1, 2, 3, 7)):
        super().__init__()
        if not scales or any(s < 1 for s in scales):
            raise ValueError(f"scales must be positive integers, got {scales!r}")
        self.scales = tuple(int(s) for s in scales)

    @property
    def num_tokens(self) -> int:
        """`Nm` produced by this configuration."""
        return sum(s * s for s in self.scales)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """[B, Np, D] on a square grid -> [B, Nm, D].

        `Np` must be a perfect square (196 -> 14x14 for ViT-B/16 at 224px).
        """
        b, n, d = patch_tokens.shape
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(
                f"patch count {n} is not a perfect square; cannot form a spatial grid. "
                "ViT-B/16 at 224px yields 196 = 14x14 patches (CLS already removed)."
            )
        # [B, Np, D] -> [B, D, H, W] for pooling
        grid = patch_tokens.transpose(1, 2).reshape(b, d, side, side)
        pooled = [
            F.adaptive_avg_pool2d(grid, output_size=s).flatten(2).transpose(1, 2)
            for s in self.scales
        ]
        return torch.cat(pooled, dim=1)  # [B, Nm, D]
