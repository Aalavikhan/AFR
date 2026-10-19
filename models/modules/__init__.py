"""
models/modules
--------------
AFR-Net research modules.

Provenance of each component — see docs/DOVE_SPEC.md and docs/MODULE_SPEC.md:

| Component                | Source                                           |
|--------------------------|--------------------------------------------------|
| `GatedSelfAttention`     | DOVE Eq (6)-(11), faithful transcription         |
| `ROAMMed` (IFA/IGA)      | DOVE Eq (17)-(28), faithful; inputs adapted      |
| `BidirectionalTripletRankingLoss` | DOVE Eq (30), faithful                  |
| `dove_total_loss`        | DOVE Eq (31), faithful                           |
| `GTGA`                   | **custom**; DTGA Eq (13)-(16) skeleton, depthwise |
| `DDTP`                   | **custom**; no DOVE counterpart                  |
| `OTAlign`                | **custom**; no DOVE counterpart                  |
| `SpatialPyramidPool`     | **adaptation**; substitutes DOVE's MSV encoder   |
"""

from ._common import (
    ResidualMLP,
    SpatialPyramidPool,
    TwoLayerFC,
    as_bool_mask,
    masked_mean,
    zero_invalid,
)
from .ddtp import DDTP, DDTPOutput
from .gated_attention import GatedSelfAttention
from .gtga import GTGA, DepthwiseSeparableBranch
from .losses import (
    AFRNetObjective,
    BidirectionalTripletRankingLoss,
    ObjectiveOutput,
    clip_contrastive_loss,
    dove_total_loss,
)
from .ot_align import OTAlign, OTOutput, log_sinkhorn
from .roam_med import (
    InterModalGuidanceAttention,
    IntraModalFusionAttention,
    ROAMMed,
    ROAMOutput,
)

__all__ = [
    "AFRNetObjective",
    "BidirectionalTripletRankingLoss",
    "DDTP",
    "DDTPOutput",
    "DepthwiseSeparableBranch",
    "GTGA",
    "GatedSelfAttention",
    "InterModalGuidanceAttention",
    "IntraModalFusionAttention",
    "OTAlign",
    "OTOutput",
    "ObjectiveOutput",
    "ROAMMed",
    "ROAMOutput",
    "ResidualMLP",
    "SpatialPyramidPool",
    "TwoLayerFC",
    "as_bool_mask",
    "clip_contrastive_loss",
    "dove_total_loss",
    "log_sinkhorn",
    "masked_mean",
    "zero_invalid",
]
