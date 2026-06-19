"""
scripts/download_roco.py
------------------------
Downloads ROCOv2-radiology dataset AND BiomedCLIP model weights
directly into your project directory. Nothing goes to system cache.

Directory layout after running:
    afrnet/
      data/
        roco/
          train/        ← ~60k samples (save_to_disk format)
          validation/   ← ~9.9k samples
          test/         ← ~9.9k samples
        hf_cache/       ← temporary HF download cache (safe to delete after)
      models/
        biomedclip/     ← BiomedCLIP weights (open_clip local load)

Run:
    uv run python scripts/download_roco.py
Re-running is safe — skips already-completed splits.
"""

import os
import sys
import time
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "roco"
HF_CACHE    = ROOT / "data" / "hf_cache"   # temp download cache, local to project
MODEL_DIR   = ROOT / "models" / "biomedclip"

HF_DATASET  = "eltorio/ROCOv2-radiology"
HF_MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
SPLITS      = ["train", "validation", "test"]
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_hf_env():
    """Point ALL HuggingFace downloads to the local project cache."""
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]          = str(HF_CACHE)
    os.environ["HF_DATASETS_CACHE"]= str(HF_CACHE / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["TORCH_HOME"]       = str(HF_CACHE / "torch")
    print(f"ℹ  HF cache → {HF_CACHE}")


def download_split(load_dataset, split: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{MAX_RETRIES} ...")
            ds = load_dataset(
                HF_DATASET,
                split=split,
                cache_dir=str(HF_CACHE / "datasets"),
            )
            return ds
        except Exception as e:
            print(f"  ⚠  Attempt {attempt} failed: {str(e)[:120]}")
            if attempt < MAX_RETRIES:
                wait = attempt * 15
                print(f"     Retrying in {wait}s (completed shards are kept) ...")
                time.sleep(wait)
            else:
                print(f"\n❌ All {MAX_RETRIES} retries failed for '{split}'.")
                print("   Re-run the script — completed shards are preserved.")
                sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset():
    print("\n" + "=" * 55)
    print("STEP 1 — Dataset: ROCOv2-radiology (~6-8 GB)")
    print("=" * 55)

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 'datasets' not installed.  Fix: uv add datasets")
        sys.exit(1)

    for split in SPLITS:
        out_dir   = DATA_DIR / split
        done_flag = out_dir / ".done"

        if done_flag.exists():
            print(f"✅ {split:12s} already saved — skipping")
            continue

        print(f"\nDownloading split: {split} ...")
        ds = download_split(load_dataset, split)

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving {len(ds):,} samples → {out_dir}")
        ds.save_to_disk(str(out_dir))
        done_flag.touch()

        sample  = ds[0]
        caption = sample.get("caption", "")
        print(f"  ✅ {split}: {len(ds):,} samples")
        print(f"     Columns : {list(sample.keys())}")
        print(f"     Sample  : '{str(caption)[:80]}'")

    print(f"\n✅ Dataset saved to {DATA_DIR}")


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model():
    print("\n" + "=" * 55)
    print("STEP 2 — Model: BiomedCLIP weights (~1.5 GB)")
    print("=" * 55)

    done_flag = MODEL_DIR / ".done"
    if done_flag.exists():
        print(f"✅ BiomedCLIP already saved at {MODEL_DIR} — skipping")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 'huggingface_hub' not installed.  Fix: uv add huggingface_hub")
        sys.exit(1)

    print(f"Downloading {HF_MODEL_ID} → {MODEL_DIR}")
    print("(This is ~1.5 GB — downloads once)\n")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=HF_MODEL_ID,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,   # real files, no symlinks
                cache_dir=str(HF_CACHE / "hub"),
            )
            done_flag.touch()
            print(f"\n✅ BiomedCLIP saved to {MODEL_DIR}")
            return
        except Exception as e:
            print(f"  ⚠  Attempt {attempt}/{MAX_RETRIES} failed: {str(e)[:120]}")
            if attempt < MAX_RETRIES:
                wait = attempt * 15
                print(f"     Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print("❌ Model download failed after all retries.")
                sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("AFR-Net — Local Download Script")
    print("Everything saved inside your project folder.")
    print("=" * 55)

    set_hf_env()

    download_dataset()
    download_model()

    print("\n" + "=" * 55)
    print("✅ All downloads complete.")
    print()
    print(f"  Dataset : {DATA_DIR}")
    print(f"  Model   : {MODEL_DIR}")
    print(f"  HF cache: {HF_CACHE}  ← safe to delete after training")
    print()
    print("Next steps:")
    print("  1. uv run python train.py --smoke   (verify pipeline)")
    print("  2. uv run python train.py           (full baseline run)")
    print("=" * 55)


if __name__ == "__main__":
    main()
