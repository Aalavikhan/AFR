"""
train.py
---------
Entry point for AFR-Net baseline training on ROCO.

Usage:
    uv run python train.py                        # full training run
    uv run python train.py --config configs/baseline.yaml
    uv run python train.py --smoke                # 1 epoch, 200 samples — pipeline check
    uv run python train.py --eval-only            # load best.pt and evaluate
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/baseline.yaml")
    p.add_argument("--smoke",     action="store_true", help="Quick pipeline check")
    p.add_argument("--eval-only", action="store_true", help="Evaluate best checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Smoke-test overrides
    if args.smoke:
        print("🔥 SMOKE TEST — 1 epoch, 200 samples, batch 8")
        config["training"]["epochs"]      = 1
        config["training"]["batch_size"]  = 8
        config["training"]["num_workers"] = 0
        config["data"]["max_samples"]     = 200

    # --- Device ---
    if not torch.cuda.is_available():
        print("⚠  CUDA not available — running on CPU (very slow)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        gpu  = torch.cuda.get_device_properties(0)
        vram = gpu.total_memory / 1e9
        print(f"GPU: {gpu.name} ({vram:.1f} GB VRAM)")

    # --- Model ---
    # Resolve local model directory (relative to project root)
    local_dir = config["model"].get("local_dir", None)
    if local_dir:
        local_dir = Path(local_dir)
        if not local_dir.is_absolute():
            local_dir = Path(__file__).parent / local_dir

    print("\nLoading BiomedCLIP ...")
    from models.baseline import BiomedCLIPBaseline
    model = BiomedCLIPBaseline(
        local_model_dir=local_dir,
        freeze_backbone=config["model"].get("freeze_backbone", False),
    )

    # --- Data ---
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        data_root = Path(__file__).parent / data_root

    if not data_root.exists():
        print(f"\n❌ ROCO data not found at '{data_root}'")
        print("   Run the download script first:")
        print("   uv run python scripts\\download_roco.py")
        sys.exit(1)

    print("Building data loaders ...")
    from data.roco_dataset import build_roco_loaders
    train_loader, val_loader = build_roco_loaders(
        config=config,
        tokenizer=model.tokenizer,
        preprocess_train=model.preprocess_train,
        preprocess_val=model.preprocess_val,
    )

    # --- Trainer ---
    from trainers.baseline_trainer import BaselineTrainer
    trainer = BaselineTrainer(model, train_loader, val_loader, config, device)

    if args.eval_only:
        print("\nEval-only — loading best.pt ...")
        trainer.load_checkpoint("best.pt")
        from evaluators.retrieval_eval import RetrievalEvaluator
        RetrievalEvaluator(model, device).evaluate(val_loader)
        return

    # --- Train ---
    trainer.train()

    # --- Final eval ---
    print("\nFinal evaluation on best checkpoint ...")
    trainer.load_checkpoint("best.pt")
    from evaluators.retrieval_eval import RetrievalEvaluator
    metrics = RetrievalEvaluator(model, device).evaluate(val_loader)

    print("\n" + "=" * 45)
    print("  BASELINE RESULTS  —  record these carefully")
    print("=" * 45)
    for k, v in metrics.items():
        fmt = f"{v:.1f}" if "rank" in k else f"{v:.4f}"
        print(f"  {k:<22} : {fmt}")
    print("=" * 45)
    print("\nBaseline locked. Ready for AFR-Net modules (Segment 3).")


if __name__ == "__main__":
    main()
