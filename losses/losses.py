"""
losses/losses.py
----------------
Combined AFR-Net objective:

    L_total = L_SDM  +  alpha(step) * L_OT  +  beta * L_distill

* L_SDM     — symmetric contrastive (InfoNCE) on the AFR-Net global features
              (v_final from DDTP  vs  t_final pooled from GTGA). This is the
              term the retrieval metrics track.
* L_OT      — fine-grained transport cost from OT-Align. Ramped in with a warmup
              (weight 0 for the first `warmup_steps`) — plan risk item: enabling
              OT from step 0 destabilises a cold DDTP.
* L_distill — optional cosine pull of AFR-Net globals toward the FROZEN BiomedCLIP
              globals, so fine-tuning does not forget the pretrained space.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(v: torch.Tensor, t: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric CLIP loss on L2-normalised globals. v,t : [B, D]."""
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * v @ t.t()                         # [B, B]
    labels = torch.arange(v.shape[0], device=v.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@dataclass
class LossConfig:
    ot_weight: float = 0.1
    ot_warmup_steps: int = 1000
    distill_weight: float = 0.0     # set >0 to enable forgetting-resistance


class AFRLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        # own learnable temperature for the AFR-Net contrastive space
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    def ot_alpha(self, step: int) -> float:
        if step >= self.cfg.ot_warmup_steps:
            return self.cfg.ot_weight
        return self.cfg.ot_weight * (step / max(1, self.cfg.ot_warmup_steps))

    def forward(
        self,
        v_final: torch.Tensor,          # [B, D]
        t_final: torch.Tensor,          # [B, D]
        L_OT: torch.Tensor,             # scalar
        step: int,
        v_cls_frozen: torch.Tensor = None,   # [B, D] frozen BiomedCLIP image global
        t_frozen: torch.Tensor = None,       # [B, D] frozen BiomedCLIP text  global
    ):
        L_sdm = info_nce(v_final, t_final, self.logit_scale)

        alpha = self.ot_alpha(step)
        L_total = L_sdm + alpha * L_OT

        L_distill = torch.tensor(0.0, device=v_final.device)
        if self.cfg.distill_weight > 0 and v_cls_frozen is not None:
            d_img = 1.0 - F.cosine_similarity(
                F.normalize(v_final, dim=-1), F.normalize(v_cls_frozen, dim=-1), dim=-1
            ).mean()
            d_txt = 1.0 - F.cosine_similarity(
                F.normalize(t_final, dim=-1), F.normalize(t_frozen, dim=-1), dim=-1
            ).mean()
            L_distill = 0.5 * (d_img + d_txt)
            L_total = L_total + self.cfg.distill_weight * L_distill

        parts = {
            "L_sdm": L_sdm.detach(),
            "L_ot": L_OT.detach(),
            "L_distill": L_distill.detach(),
            "ot_alpha": alpha,
        }
        return L_total, parts
