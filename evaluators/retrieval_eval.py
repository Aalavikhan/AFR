"""
evaluators/retrieval_eval.py
-----------------------------
Production retrieval evaluator: R@1/5/10 and MRR for image-to-text
and text-to-image retrieval.

Designed to run on full validation/test sets efficiently:
  - Encodes all images and texts in batches (no OOM on large sets)
  - Computes similarity matrix on GPU if available
  - Returns a flat metrics dict ready for wandb/logging

Usage:
    from evaluators.retrieval_eval import RetrievalEvaluator
    evaluator = RetrievalEvaluator(model, device)
    metrics = evaluator.evaluate(val_loader)
    # {'i2t_R@1': 0.42, 'i2t_R@5': 0.71, ..., 'i2t_MRR': 0.55, ...}
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class RetrievalEvaluator:
    """
    Encodes an entire DataLoader into feature matrices, then computes
    retrieval metrics without loading the model again.

    Args:
        model:   BiomedCLIPBaseline (or any model with encode_image/encode_text)
        device:  torch device
        ks:      R@K values to compute
    """

    def __init__(self, model, device: torch.device, ks: Tuple[int, ...] = (1, 5, 10)):
        self.model  = model
        self.device = device
        self.ks     = ks

    @torch.no_grad()
    def encode_dataset(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the full DataLoader through the model.

        Returns:
            img_feats: [N, D]  normalized
            txt_feats: [N, D]  normalized
        """
        self.model.eval()
        all_img, all_txt = [], []

        for images, tokens in tqdm(loader, desc="Encoding", leave=False):
            images = images.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            img_f = self.model.encode_image(images)   # already normalized
            txt_f = self.model.encode_text(tokens)

            all_img.append(img_f.cpu())
            all_txt.append(txt_f.cpu())

        img_feats = torch.cat(all_img, dim=0)  # [N, D]
        txt_feats = torch.cat(all_txt, dim=0)  # [N, D]
        return img_feats, txt_feats

    def compute_metrics(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute R@K and MRR from pre-encoded feature matrices.

        Args:
            img_feats: [N, D] normalized (CPU or GPU)
            txt_feats: [N, D] normalized (CPU or GPU)
        Returns:
            dict of metric_name -> float
        """
        # Move to GPU for fast matmul if possible
        compute_dev = self.device if self.device.type == "cuda" else torch.device("cpu")
        img_feats = img_feats.to(compute_dev)
        txt_feats = txt_feats.to(compute_dev)

        N = img_feats.shape[0]
        sim = img_feats @ txt_feats.T   # [N, N]
        gt  = torch.arange(N, device=compute_dev)

        results = {}

        for k in self.ks:
            k_eff = min(k, N)

            # image → text
            topk_i2t = sim.topk(k_eff, dim=1).indices           # [N, k]
            hits_i2t = (topk_i2t == gt.unsqueeze(1)).any(dim=1).float().mean().item()
            results[f"i2t_R@{k}"] = hits_i2t

            # text → image
            topk_t2i = sim.T.topk(k_eff, dim=1).indices         # [N, k]
            hits_t2i = (topk_t2i == gt.unsqueeze(1)).any(dim=1).float().mean().item()
            results[f"t2i_R@{k}"] = hits_t2i

        # MRR
        # rank of the correct item (1-indexed)
        sorted_i2t = sim.argsort(dim=1, descending=True)        # [N, N]
        rank_i2t   = (sorted_i2t == gt.unsqueeze(1)).float().argmax(dim=1) + 1
        results["i2t_MRR"] = (1.0 / rank_i2t.float()).mean().item()

        sorted_t2i = sim.T.argsort(dim=1, descending=True)
        rank_t2i   = (sorted_t2i == gt.unsqueeze(1)).float().argmax(dim=1) + 1
        results["t2i_MRR"] = (1.0 / rank_t2i.float()).mean().item()

        # Median rank (useful diagnostic)
        results["i2t_median_rank"] = float(rank_i2t.float().median().item())
        results["t2i_median_rank"] = float(rank_t2i.float().median().item())

        return results

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Full evaluation pipeline: encode → compute metrics.

        Returns:
            dict of metric_name -> float
        """
        img_feats, txt_feats = self.encode_dataset(loader)
        metrics = self.compute_metrics(img_feats, txt_feats)

        # Pretty print
        print("\n--- Retrieval Metrics ---")
        for k, v in metrics.items():
            if "rank" in k:
                print(f"  {k:<25}: {v:.1f}")
            else:
                print(f"  {k:<25}: {v:.4f}")
        print("-------------------------\n")

        return metrics
