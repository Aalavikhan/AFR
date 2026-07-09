"""
models/afrnet.py
----------------
AFR-Net assembly. Wires the frozen BiomedCLIP backbone through
ROAM -> GTGA -> DDTP -> OT-Align.

Two interfaces
--------------
1. Retrieval interface (used by evaluators/retrieval_eval.py, UNCHANGED):
       encode_image(images) -> [B, D] normalised   (text-free)
       encode_text(tokens)  -> [B, D] normalised   (image-free)
   These are DECOUPLED on purpose so the standard O(N) evaluator is valid and
   the features it ranks are exactly the ones L_SDM trains (no train/eval gap).

2. Training interface:
       forward(images, tokens) -> dict with retrieval globals + L_OT + aux
   The COUPLED path (ROAM IGA with real region descriptor, DDTP with real text
   summary) feeds OT-Align as auxiliary fine-grained supervision; this is what
   trains DDTP. DDTP's own text-conditional global and two-stage re-ranking are
   the documented faithful-eval upgrade, deferred for v1.

Backbone contract (BiomedCLIPBaseline provides all of these):
    get_patch_tokens(images) -> [B, 196, 768]
    get_word_tokens(tokens)  -> [B, 256, 768]
    encode_image(images)     -> [B, 512] normalised   (frozen global)
    encode_text(tokens)      -> [B, 512] normalised   (frozen global)
    embed_dim                -> int
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.roam import ROAMMed
from models.gtga import GTGA
from models.ddtp import DDTP
from models.ot_align import OTAlign


class AFRNet(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_dim: int = 768,     # backbone token dim
        proj_dim: int = 512,       # shared retrieval space
        pad_token_id: int = 0,
        keep_ratio: float = 0.5,
        num_patches: int = 196,
    ):
        super().__init__()
        self.backbone = backbone
        self.pad_token_id = pad_token_id

        self.roam = ROAMMed(dim=hidden_dim, num_heads=8)
        self.gtga = GTGA(dim=hidden_dim, num_heads=8)
        self.ddtp = DDTP(dim=hidden_dim, keep_ratio=keep_ratio, num_patches=num_patches)
        self.ot   = OTAlign(eps=0.05, n_iters=50)

        # projection heads into the shared retrieval space
        self.img_proj = nn.Linear(hidden_dim, proj_dim)
        self.txt_proj = nn.Linear(hidden_dim, proj_dim)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _text_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """True = valid (non-PAD) token. [B, L]"""
        return tokens != self.pad_token_id

    def _pool_visual(self, V_MR: torch.Tensor) -> torch.Tensor:
        """Mean+max pool patches -> [B, D]."""
        return 0.5 * (V_MR.mean(dim=1) + V_MR.max(dim=1).values)

    def _pool_text(self, F_G: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean over valid tokens -> [B, D]."""
        m = mask.unsqueeze(-1).float()
        return (F_G * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    # ------------------------------------------------------------------
    # retrieval interface (decoupled)
    # ------------------------------------------------------------------
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        V = self.backbone.get_patch_tokens(images)                  # [B, N, D]
        V_MR, _, _ = self.roam.visual_branch(V)                     # text-free
        g = self.img_proj(self._pool_visual(V_MR))
        return F.normalize(g, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        W = self.backbone.get_word_tokens(tokens)                   # [B, L, D]
        mask = self._text_mask(tokens)
        T_RG = self.roam.text_branch(W, r_bar=None)                 # image-free (neutral protos)
        F_G = self.gtga(T_RG, key_padding_mask=~mask)
        g = self.txt_proj(self._pool_text(F_G, mask))
        return F.normalize(g, dim=-1)

    # ------------------------------------------------------------------
    # training interface (retrieval globals + coupled OT aux)
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        mask = self._text_mask(tokens)

        # shared backbone tokens
        V = self.backbone.get_patch_tokens(images)                  # [B, N, D]
        W = self.backbone.get_word_tokens(tokens)                   # [B, L, D]

        # frozen globals (for optional distill + DDTP CLS reference)
        with torch.no_grad():
            v_cls = self.backbone.encode_image(images)              # [B, proj] normalised
            t_eos = self.backbone.encode_text(tokens)               # [B, proj] normalised
        # project frozen globals into hidden dim space for DDTP reference use
        # (DDTP only needs a per-sample reference vector; we reuse v_cls pooled patch)
        v_cls_hidden = self._pool_visual(V)                         # [B, D] text-free ref

        # ---- retrieval globals (decoupled == eval features) ----
        V_MR, _, r_bar = self.roam.visual_branch(V, v_cls=v_cls_hidden)
        v_img = F.normalize(self.img_proj(self._pool_visual(V_MR)), dim=-1)

        T_RG_neutral = self.roam.text_branch(W, r_bar=None)
        F_G_neutral = self.gtga(T_RG_neutral, key_padding_mask=~mask)
        t_txt = F.normalize(self.txt_proj(self._pool_text(F_G_neutral, mask)), dim=-1)

        # ---- coupled path -> OT auxiliary (trains DDTP) ----
        w_eos_hidden = self._pool_text(F_G_neutral, mask)           # [B, D] text summary
        v_final, V_tilde, patch_mask, aux = self.ddtp(V_MR, w_eos_hidden, v_cls_hidden)
        T_RG_coupled = self.roam.text_branch(W, r_bar=r_bar)
        F_G_coupled = self.gtga(T_RG_coupled, key_padding_mask=~mask)
        L_OT, P_star = self.ot(V_tilde, F_G_coupled, text_mask=mask)

        return {
            "v_img": v_img,          # retrieval image global (normalised)
            "t_txt": t_txt,          # retrieval text global (normalised)
            "L_OT": L_OT,            # auxiliary transport cost
            "v_cls_frozen": v_cls,   # for optional distill
            "t_frozen": t_eos,       # for optional distill
            "P_star": P_star,        # for viz
            "patch_mask": patch_mask,
        }
