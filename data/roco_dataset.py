"""
data/roco_dataset.py
--------------------
The file train.py imports but that was missing from the repo. Loads the
ROCOv2-radiology splits already saved to disk by scripts/download_roco.py
(HuggingFace `save_to_disk` format) and returns (image, tokens) batches.

Subset control (your "prove it works on a portion" lever):
    config["data"]["max_samples"]  -> caps BOTH train and val via ds.select(...)

Batch item: (image_tensor [3,224,224], token_ids [context_length])
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_from_disk


def _to_pil(x) -> Image.Image:
    """ROCO image column may be a PIL.Image already, or a dict with bytes/path."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict):
        if x.get("bytes") is not None:
            import io
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        if x.get("path"):
            return Image.open(x["path"]).convert("RGB")
    raise TypeError(f"Unrecognised image field type: {type(x)}")


class ROCODataset(Dataset):
    def __init__(self, split_dir, tokenizer, preprocess,
                 image_key="image", caption_key="caption",
                 max_samples: Optional[int] = None):
        self.ds = load_from_disk(str(split_dir))
        if max_samples is not None:
            n = min(max_samples, len(self.ds))
            self.ds = self.ds.select(range(n))
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        # resolve column names defensively
        cols = set(self.ds.column_names)
        self.image_key = image_key if image_key in cols else _first_present(
            cols, ["image", "img", "jpg"])
        self.caption_key = caption_key if caption_key in cols else _first_present(
            cols, ["caption", "text", "report", "findings"])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = self.preprocess(_to_pil(sample[self.image_key]))       # [3,224,224]
        caption = str(sample[self.caption_key])
        tokens = self.tokenizer([caption])[0]                        # [context_length]
        return img, tokens


def _first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of {candidates} in dataset columns {sorted(cols)}")


def build_roco_loaders(config, tokenizer, preprocess_train, preprocess_val):
    root = Path(config["data"]["root"])
    if not root.is_absolute():
        root = Path(__file__).parent.parent / root
    max_samples = config["data"].get("max_samples", None)
    tcfg = config["training"]

    train_ds = ROCODataset(root / "train", tokenizer, preprocess_train,
                           max_samples=max_samples)
    # cap val at a modest size for fast, stable retrieval metrics on subsets
    val_cap = config["data"].get("val_max_samples", max_samples)
    val_ds = ROCODataset(root / "validation", tokenizer, preprocess_val,
                         max_samples=val_cap)

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.get("eval_batch_size", tcfg["batch_size"]),
        shuffle=False, num_workers=tcfg.get("num_workers", 4), pin_memory=True)

    print(f"[ROCO] train={len(train_ds):,}  val={len(val_ds):,}  "
          f"(max_samples={max_samples})")
    return train_loader, val_loader
