"""
models/baseline.py
------------------
BiomedCLIP wrapper — loads weights from local directory.

The state dict uses visual.trunk.* and text.transformer.* keys,
which means open_clip must be initialised with the full HF config
(not plain 'ViT-B-16'). We create the architecture from the HF hub
config (cached, no download) then load the local .bin directly.

Local layout:
    models/biomedclip/
        open_clip_config.json
        open_clip_pytorch_model.bin   <- 784 MB, 352 keys
        tokenizer.json
        tokenizer_config.json
        vocab.txt
        special_tokens_map.json
"""

from __future__ import annotations

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from pathlib import Path

HF_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BiomedCLIPBaseline(nn.Module):
    """
    Thin wrapper around BiomedCLIP.

    Loading strategy (local weights):
      1. Create the model architecture using the HF hub ID — this reads the
         open_clip_config.json from your local HF cache (~/.cache/huggingface).
         It does NOT download the 784 MB weights file again.
      2. Immediately overwrite the randomly-initialised weights by loading
         your local open_clip_pytorch_model.bin with load_state_dict().
      This is the only reliable way because the state dict keys
      (visual.trunk.*, text.transformer.*) only match the TimmModel
      architecture defined in the HF config, not any built-in arch name.

    Args:
        local_model_dir : folder containing open_clip_pytorch_model.bin
                          and tokenizer files.
        freeze_backbone : freeze all params except logit_scale.
    """

    def __init__(
        self,
        local_model_dir: str | Path | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        local = Path(local_model_dir).resolve() if local_model_dir else None
        weights_path = local / "open_clip_pytorch_model.bin" if local else None
        use_local = local is not None and weights_path.exists()

        # ── Step 1: create architecture from HF config ──────────────────
        # This reads open_clip_config.json from HF cache (already there from
        # your earlier test_3_biomedclip.py run — it's tiny, <1 KB).
        # It does NOT re-download the 784 MB weights.
        print(f"[BiomedCLIPBaseline] Building architecture from HF config...")
        (self.model,
         self.preprocess_train,
         self.preprocess_val) = open_clip.create_model_and_transforms(HF_MODEL_ID)

        self.tokenizer = open_clip.get_tokenizer(HF_MODEL_ID)

        # ── Step 2: overwrite weights from local .bin ────────────────────
        if use_local:
            print(f"[BiomedCLIPBaseline] Loading weights from: {weights_path}")
            print(f"[BiomedCLIPBaseline] File size: "
                  f"{weights_path.stat().st_size / 1e6:.0f} MB")

            state_dict = torch.load(
                str(weights_path),
                map_location="cpu",
                weights_only=False,
            )

            # open_clip bins are sometimes wrapped under a 'state_dict' key
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            state_dict.pop("text.transformer.embeddings.position_ids", None)

            missing, unexpected = self.model.load_state_dict(
                state_dict, strict=True
)

            if missing:
                print(f"[BiomedCLIPBaseline] WARNING missing keys  ({len(missing)}): "
                      f"{missing[:3]}")
            if unexpected:
                print(f"[BiomedCLIPBaseline] WARNING unexpected keys ({len(unexpected)}): "
                      f"{unexpected[:3]}")
            if not missing and not unexpected:
                print(f"[BiomedCLIPBaseline] Weights loaded perfectly "
                      f"({len(state_dict)} keys matched)")
        else:
            if local_model_dir:
                print(f"[BiomedCLIPBaseline] WARNING: '{weights_path}' not found.")
            print(f"[BiomedCLIPBaseline] Using weights from HF cache.")

        # ── Expose embed dim ─────────────────────────────────────────────
        with torch.no_grad():
            feat = self.model.encode_image(torch.zeros(1, 3, 224, 224))
        self.embed_dim: int = feat.shape[-1]
        print(f"[BiomedCLIPBaseline] Ready — embed_dim={self.embed_dim}")

        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, D]  L2-normalised"""
        return F.normalize(self.model.encode_image(images), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, seq_len] -> [B, D]  L2-normalised"""
        return F.normalize(self.model.encode_text(tokens), dim=-1)

    def tokenize(self, texts) -> torch.Tensor:
        """Tokenise a list of strings -> [B, seq_len] on CPU"""
        return self.tokenizer(texts)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def clip_loss(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric CLIP cross-entropy loss."""
        logit_scale = self.model.logit_scale.exp().clamp(max=100.0)
        logits_i2t  = logit_scale * img_feats @ txt_feats.T
        labels      = torch.arange(img_feats.shape[0], device=img_feats.device)
        loss_i2t    = F.cross_entropy(logits_i2t,   labels)
        loss_t2i    = F.cross_entropy(logits_i2t.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

    # ------------------------------------------------------------------
    # Patch / word token access  (used by AFR-Net modules later)
    # ------------------------------------------------------------------

    def get_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        ViT patch tokens BEFORE global pooling.
        Returns [B, 196, 768]  (14x14 patches, 768-dim each).
        """
        all_tokens = self.model.visual.trunk.forward_features(images)
        return all_tokens[:, 1:, :]   # drop CLS token at position 0

    def get_word_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        PubMedBERT word tokens BEFORE global pooling.
        Returns [B, 256, 768].
        tokens : [B, 256] token IDs from self.tokenizer()
        """
        emb = self.model.text.token_embedding(tokens)
        emb = emb + self.model.text.positional_embedding
        out = self.model.text.transformer(emb.permute(1, 0, 2))
        return out.permute(1, 0, 2)   # [B, 256, 768]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if name != "logit_scale":
                param.requires_grad_(False)
        print("[BiomedCLIPBaseline] Backbone frozen (logit_scale trainable only).")

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad_(True)
        print("[BiomedCLIPBaseline] Backbone unfrozen.")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self):
        total     = self.total_params()
        trainable = self.trainable_params()
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}")
        print(f"  Frozen params    : {total - trainable:,}")