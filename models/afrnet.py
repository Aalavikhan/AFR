"""
models/afrnet.py
-----------------
AFRNet — composes the frozen BiomedCLIP backbone with the four research
modules: GTGA, DDTP, ROAM-Med and OT-Align.

Forward pass
------------
    images [B,3,224,224]                tokens [B,L]
          |                                   |
          | backbone.get_patch_tokens         | backbone.encode_word_tokens
          v  (frozen, no_grad)                v  (frozen, no_grad)
    patches [B,196,768]                 words [B,L,768] + mask [B,L]
          |                                   |
          |                                   v  GTGA (depthwise, custom)
          |                             F_G [B,L,768]
          |                                   |
          |                                   v  masked mean
          |                             text_query [B,768]
          |                                   |
          +--------> DDTP(patches, text_query) <-- text-conditioned selection
          |             |
          |             v
          |          F_R [B,98,768] + region scores/mask
          |
          v  SpatialPyramidPool                      (adaptation of MSV encoder)
    F_M [B,Nm,768]
          |
          +--> ROAM-Med: IFA(F_M, F_R, prior=scores)  -> F_MR [B,Nm+98,768]
                         IGA(F_R, F_G)                -> F_RG [B,768]
          |
          v  Eq (29) means + optional projection
    V_MR, T_RG   (retrieval embeddings)
    V_M,  T_G    (global constraint embeddings, DOVE Eq 31)
    L_OT = OTAlign(F_R, F_G)                          (auxiliary loss)

Training objective
------------------
    L_total = L_primary(V_MR, T_RG) + lambda_ot * L_OT + lambda_g * L(V_M, T_G)

`L_primary` defaults to the **existing** BiomedCLIP symmetric InfoNCE so the
baseline objective is preserved; see `models/modules/losses.py`.

Frozen backbone
---------------
`AFRNet.__init__` calls `backbone.freeze_all()` and extracts backbone features
under `torch.no_grad()` by default. Only the four modules (plus optional
projection heads) are trainable. `trainable_parameters()` returns exactly those
and is what the optimizer should be built from.

Segmentor
---------
This model never constructs, imports or calls a segmentor. Segmentation-derived
information enters **only** through the optional `region_prior` /
`patch_mask` arguments of `forward`. See docs/SEGMENTOR_INTEGRATION.md.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn

from .modules import (
    DDTP,
    GTGA,
    OTAlign,
    ROAMMed,
    SpatialPyramidPool,
    masked_mean,
)


class AFRNetOutput(NamedTuple):
    v_final: torch.Tensor                 # [B, E] visual retrieval embedding (V_MR)
    t_final: torch.Tensor                 # [B, E] textual retrieval embedding (T_RG)
    v_global: torch.Tensor                # [B, E] V_M, for the Eq (31) constraint
    t_global: torch.Tensor                # [B, E] T_G, for the Eq (31) constraint
    ot_loss: Optional[torch.Tensor]       # scalar or None
    aux: Dict[str, Any]                   # intermediates for tests / inspection


class AFRNet(nn.Module):
    """Frozen BiomedCLIP + GTGA + DDTP + ROAM-Med + OT-Align.

    Args:
        backbone: a `BiomedCLIPBaseline`. Frozen in `__init__`.
        dim: token feature size exposed by the backbone (768).
        embed_dim: retrieval embedding size. `None` keeps `dim` (no projection).
        pyramid_scales: `SpatialPyramidPool` scales -> `Nm`.
        ddtp_ratio / ddtp_num_keep: selection size. 0.5 over 196 gives 98.
        ot_enabled: compute the OT-Align loss in `forward`.
        ot_source: which visual tokens enter OT — "regions" (DDTP output,
            default, cheaper and text-aligned) or "patches" (all 196).
        freeze_backbone_grad: wrap backbone feature extraction in
            `torch.no_grad()`. True by default; set False only if the backbone
            is intentionally being tuned.
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim: int = 768,
        embed_dim: Optional[int] = None,
        # --- GTGA ---
        gtga_kernel_sizes: Sequence[int] = (3, 5),
        gtga_branch_mode: str = "symmetric",
        gtga_use_eq8_gate: bool = True,
        gtga_use_gated_self_attention: bool = False,
        gtga_probe_activation: str = "sigmoid",
        # --- DDTP ---
        ddtp_ratio: float = 0.5,
        ddtp_num_keep: Optional[int] = None,
        ddtp_tau: float = 0.07,
        ddtp_score_weighting: str = "softmax_norm",
        # --- ROAM-Med ---
        pyramid_scales: Sequence[int] = (1, 2, 3, 7),
        roam_prior_mode: str = "multiply",
        roam_iga_mode: str = "batch",
        use_region_prior: bool = True,
        # --- OT-Align ---
        ot_enabled: bool = True,
        ot_source: str = "regions",
        ot_eps: float = 0.05,
        ot_max_iters: int = 100,
        ot_detach_plan: bool = True,
        # --- misc ---
        dropout: float = 0.0,
        freeze_backbone_grad: bool = True,
    ):
        super().__init__()
        if ot_source not in ("regions", "patches"):
            raise ValueError(f"ot_source must be regions/patches, got {ot_source!r}")

        self.backbone = backbone
        self.dim = dim
        self.use_region_prior = use_region_prior
        self.ot_enabled = ot_enabled
        self.ot_source = ot_source
        self.freeze_backbone_grad = freeze_backbone_grad

        # Fully freeze the backbone: AFR-Net trains only its own modules.
        if hasattr(backbone, "freeze_all"):
            backbone.freeze_all()
        else:
            for p in backbone.parameters():
                p.requires_grad_(False)

        self.gtga = GTGA(
            dim=dim,
            kernel_sizes=gtga_kernel_sizes,
            branch_mode=gtga_branch_mode,
            probe_activation=gtga_probe_activation,
            use_eq8_gate=gtga_use_eq8_gate,
            use_gated_self_attention=gtga_use_gated_self_attention,
            dropout=dropout,
        )
        self.ddtp = DDTP(
            dim=dim,
            ratio=ddtp_ratio,
            num_keep=ddtp_num_keep,
            tau=ddtp_tau,
            score_weighting=ddtp_score_weighting,
        )
        self.pyramid = SpatialPyramidPool(scales=pyramid_scales)
        self.roam = ROAMMed(
            dim=dim,
            prior_mode=roam_prior_mode,
            iga_mode=roam_iga_mode,
        )
        self.ot = (
            OTAlign(
                dim=dim, eps=ot_eps, max_iters=ot_max_iters,
                detach_plan=ot_detach_plan,
            )
            if ot_enabled
            else None
        )

        if embed_dim is None or embed_dim == dim:
            self.visual_head = nn.Identity()
            self.text_head = nn.Identity()
            self.embed_dim = dim
        else:
            self.visual_head = nn.Linear(dim, embed_dim)
            self.text_head = nn.Linear(dim, embed_dim)
            self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    @property
    def num_regions(self) -> int:
        """`Nr` for a 196-patch input (98 at the default ratio)."""
        return self.ddtp.resolve_k(196)

    @property
    def num_scales(self) -> int:
        """`Nm` produced by the pyramid."""
        return self.pyramid.num_tokens

    def trainable_parameters(self):
        """Exactly the parameters the optimizer should update."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_module_names(self):
        return sorted(
            name for name, p in self.named_parameters() if p.requires_grad
        )

    # ------------------------------------------------------------------
    def encode_backbone(self, images: torch.Tensor, tokens: torch.Tensor):
        """Frozen-backbone feature extraction.

        Returns `(patches [B,196,768], words [B,L,768], text_mask [B,L])`.
        """
        ctx = torch.no_grad() if self.freeze_backbone_grad else _NullCtx()
        with ctx:
            patches = self.backbone.get_patch_tokens(images)
            words, text_mask = self.backbone.encode_word_tokens(tokens)
        return patches, words, text_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        region_prior_external: Optional[torch.Tensor] = None,
    ) -> AFRNetOutput:
        """
        Args:
            images: [B, 3, 224, 224]
            tokens: [B, L] token ids
            patch_mask: optional [B, 196] bool, True = eligible patch. The
                documented entry point for a **separate** segmentor's
                foreground mask. Never fabricated here.
            region_prior_external: optional [B, 196] soft prior over patches,
                also a segmentor entry point. When given it is gathered at the
                DDTP-selected indices and used in place of DDTP's own scores
                as ROAM-Med's soft attention prior.
        """
        patches, words, text_mask = self.encode_backbone(images, tokens)

        if patches.shape[-1] != self.dim:
            raise ValueError(
                f"backbone patch dim {patches.shape[-1]} != configured dim {self.dim}"
            )

        # --- GTGA: contextual text -> phrase-aware word features -------------
        f_g = self.gtga(words, mask=text_mask)                       # [B, L, 768]
        text_query = masked_mean(f_g, text_mask, dim=1)              # [B, 768]

        # --- DDTP: text-conditioned region selection -------------------------
        sel = self.ddtp(patches, text_query, patch_mask=patch_mask)  # k = 98
        f_r = sel.features                                           # [B, 98, 768]

        # --- soft attention prior over the selected regions ------------------
        # `prior_is_logits` is stated explicitly rather than inferred from the
        # values: DDTP scores are cosine/tau (unbounded logits), whereas an
        # external segmentor prior is normally already a probability in [0,1].
        if not self.use_region_prior:
            prior, prior_is_logits = None, True
        elif region_prior_external is not None:
            prior = region_prior_external.gather(1, sel.indices)     # [B, 98]
            prior_is_logits = False
        else:
            prior = sel.scores                                       # [B, 98]
            prior_is_logits = True

        # --- multiscale visual tokens (MSV-encoder substitute) ---------------
        f_m = self.pyramid(patches)                                  # [B, Nm, 768]

        # --- ROAM-Med: IFA + IGA ---------------------------------------------
        roam = self.roam(
            f_m=f_m,
            f_r=f_r,
            f_g=f_g,
            region_prior=prior,
            region_mask=sel.mask,
            text_mask=text_mask,
            prior_is_logits=prior_is_logits,
        )

        # --- Eq (29) global representations ---------------------------------
        v_m = masked_mean(f_m, None, dim=1)                          # [B, 768]
        t_g = text_query                                             # [B, 768]

        v_final = self.visual_head(roam.v_mr)
        t_final = self.text_head(roam.f_rg)
        v_global = self.visual_head(v_m)
        t_global = self.text_head(t_g)

        # --- OT-Align auxiliary loss -----------------------------------------
        ot_loss = None
        ot_out = None
        if self.ot is not None:
            if self.ot_source == "regions":
                vis, vis_mask = f_r, sel.mask
            else:
                vis, vis_mask = patches, patch_mask
            ot_out = self.ot(vis, f_g, patch_mask=vis_mask, token_mask=text_mask)
            ot_loss = ot_out.loss

        aux: Dict[str, Any] = {
            "patches": patches,
            "words": words,
            "text_mask": text_mask,
            "f_g": f_g,
            "f_m": f_m,
            "f_r": f_r,
            "f_mr": roam.f_mr,
            "f_rg": roam.f_rg,
            "s_mr": roam.s_mr,
            "s_rg": roam.s_rg,
            "ddtp_indices": sel.indices,
            "ddtp_scores": sel.scores,
            "ddtp_mask": sel.mask,
            "ddtp_all_scores": sel.all_scores,
            "region_prior": prior,
            "ot": ot_out,
        }
        return AFRNetOutput(
            v_final=v_final,
            t_final=t_final,
            v_global=v_global,
            t_global=t_global,
            ot_loss=ot_loss,
            aux=aux,
        )

    # ------------------------------------------------------------------
    # Retrieval interface — matches evaluators/retrieval_eval.py
    # ------------------------------------------------------------------
    # RetrievalEvaluator calls encode_image(images) and encode_text(tokens)
    # independently. AFR-Net cannot honour that contract faithfully, because
    # DDTP conditions the visual side on the text and IGA conditions the text
    # side on the visual regions: neither embedding exists without the other.
    # Rather than silently returning something unconditioned, these raise.
    # See docs/FIDELITY_AUDIT.md WARN-EVAL-1 for the required evaluator change.

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "AFRNet embeddings are cross-conditioned (DDTP uses the text query; "
            "IGA uses the regions), so an image embedding cannot be produced "
            "without its text. Use forward(images, tokens) and see "
            "docs/FIDELITY_AUDIT.md WARN-EVAL-1."
        )

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "AFRNet embeddings are cross-conditioned; see encode_image()."
        )


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
