from datasets import load_dataset
from pathlib import Path

out = Path("data/roco")
for split in ["train", "validation", "test"]:
    print(f"loading {split} ...", flush=True)
    ds = load_dataset("eltorio/ROCOv2-radiology", split=split)
    dest = out / split
    dest.mkdir(parents=True, exist_ok=True)
    print(f"saving {split} -> {dest}  ({len(ds)} rows)", flush=True)
    ds.save_to_disk(str(dest))
    (dest / ".done").touch()
print("ALL SPLITS SAVED", flush=True)