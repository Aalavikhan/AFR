"""
train_afrnet.py
---------------
Entry point for full AFR-Net training on ROCO (no other datasets).

Usage:
    uv run python train_afrnet.py --smoke                     # pipeline check
    uv run python train_afrnet.py --config configs/afrnet_roco.yaml
    uv run python train_afrnet.py --eval-only                 # eval best.pt

Prints, at the end, AFR-Net retrieval metrics AND the frozen-BiomedCLIP
zero-shot numbers on the same val subset, so you immediately see whether the
modules helped (the whole point of the prove-it phase).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/afrnet_roco.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # force offline once data + weights are local
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "offline")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"❌ config not found: {cfg_path}"); sys.exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.smoke:
        print("🔥 SMOKE — 1 epoch, 200 samples")
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 8
        config["training"]["eval_batch_size"] = 16
        config["training"]["grad_accum"] = 1
        config["training"]["num_workers"] = 0
        config["data"]["max_samples"] = 200
        config["data"]["val_max_samples"] = 100
        config["loss"]["ot_warmup_steps"] = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} ({gpu.total_memory/1e9:.1f} GB)")

    # --- backbone ---
    local_dir = Path(config["model"]["local_dir"])
    if not local_dir.is_absolute():
        local_dir = Path(__file__).parent / local_dir
    from models.baseline import BiomedCLIPBaseline
    backbone = BiomedCLIPBaseline(local_model_dir=local_dir, freeze_backbone=True)

    # --- AFR-Net ---
    from models.afrnet import AFRNet
    model = AFRNet(
        backbone,
        hidden_dim=config["model"].get("hidden_dim", 768),
        proj_dim=config["model"].get("proj_dim", 512),
        keep_ratio=config["model"].get("keep_ratio", 0.5),
        num_patches=config["model"].get("num_patches", 196),
        pad_token_id=0,
    )

    # --- data ---
    from data.roco_dataset import build_roco_loaders
    train_loader, val_loader = build_roco_loaders(
        config, backbone.tokenizer, backbone.preprocess_train, backbone.preprocess_val)

    from trainers.afrnet_trainer import AFRNetTrainer
    trainer = AFRNetTrainer(model, train_loader, val_loader, config, device)

    from evaluators.retrieval_eval import RetrievalEvaluator

    # --- BEFORE number: frozen BiomedCLIP zero-shot on the same val subset ---
    print("\n[baseline] frozen BiomedCLIP zero-shot on val subset:")
    RetrievalEvaluator(backbone, device).evaluate(val_loader)

    if args.eval_only:
        trainer.load_checkpoint("best.pt")
        print("\n[AFR-Net] eval-only:")
        RetrievalEvaluator(model, device).evaluate(val_loader)
        return

    trainer.train()

    print("\n[AFR-Net] final eval on best checkpoint:")
    trainer.load_checkpoint("best.pt")
    metrics = RetrievalEvaluator(model, device).evaluate(val_loader)
    print("\n" + "=" * 50)
    print("  AFR-Net vs frozen BiomedCLIP — compare the R@1 above")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}" if "rank" not in k else f"  {k:<22}: {v:.1f}")


if __name__ == "__main__":
    main()
