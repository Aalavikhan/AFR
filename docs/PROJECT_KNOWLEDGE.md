# AFR-Net — Project Knowledge Base

Technical reference produced by a read-only audit on **2026-10-18** against
commit `7858ee5` ("changed toml to suit new config"), working tree clean.

Every project-specific statement below is tagged:

- **[V]** VERIFIED — directly supported by inspected files or executed commands.
- **[I]** INFERENCE — reasoned from evidence, not confirmed by execution.
- **[U]** UNKNOWN / UNVERIFIED — not established; the check needed is stated.

**Update 2026-10-18 (second pass).** The *system* interpreter turned out to
have `torch 2.13.0+cpu` and `open_clip 3.3.0` installed, which allowed genuine
execution-based verification without installing anything or touching the
network. This **corrected two earlier findings** (§8 CPU/AMP behaviour, §4.3
download scope) and **verified the evaluator's arithmetic** (§9). Still true:
no package was installed, no asset downloaded, no training run, and the project
`.venv` does not exist. See §13 for the full command record and
[VERIFICATION.md](VERIFICATION.md) for the verification inventory.

---

## 1. Project goals and current research stage

**[V]** `pyproject.toml:4` — `description = "AFR-Net medical image-text retrieval"`,
package `afr`, version `0.1.0`.

**[V]** The repository implements a **BiomedCLIP contrastive retrieval
baseline** and nothing else. There is no AFR-Net-specific module, loss, or
layer in any source file.

**[V]** `AFRNet_Windows_Baseline_Guide.docx` §9 states the intended sequence:
build the ROCO dataset class ("Segment 2"), run a smoke test, record baseline
R@1, then "begin implementing AFR-Net modules (ROAM-Med first)". `train.py:119`
echoes this: `"Baseline locked. Ready for AFR-Net modules (Segment 3)."`

**[V]** Stage: **pre-baseline.** The pipeline cannot run (§14, `AFR-001`).

**[U]** What ROAM-Med is. The name appears exactly once, in the guide, with no
definition. No paper, equations, diagram or design note exists in the repo.
*Check:* ask the user for the specification.

---

## 2. Repository structure and per-file responsibility

**[V]** Complete file inventory (excluding `.git`), sizes in bytes:

| Path | Size | Responsibility |
|---|---|---|
| `train.py` | 4119 | Entry point: config load, device pick, model/loader/trainer wiring, final eval |
| `main.py` | 87 | Placeholder stub (`print("Hello from afr!")`). Referenced by nothing |
| `configs/baseline.yaml` | 1135 | All hyperparameters and paths |
| `models/baseline.py` | 8446 | `BiomedCLIPBaseline`: encode, loss, freeze, token accessors |
| `models/__init__.py` | 0 | empty package marker |
| `trainers/baseline_trainer.py` | 9299 | `BaselineTrainer` + `cosine_schedule_with_warmup` |
| `trainers/__init__.py` | 0 | empty |
| `evaluators/retrieval_eval.py` | 4976 | `RetrievalEvaluator`: encode dataset, R@K, MRR, median rank |
| `evaluators/__init__.py` | 0 | empty |
| `scripts/download_roco.py` | 6464 | Downloads ROCOv2 splits + BiomedCLIP snapshot into the project |
| `tests/test_1_python.py` | 594 | Python version print-check |
| `tests/test_2_cuda.py` | 2171 | CUDA/VRAM print-check + GPU matmul |
| `tests/test_3_biomedclip.py` | 2361 | open_clip + BiomedCLIP load/encode (downloads weights) |
| `tests/test_4_deps.py` | 2151 | Import/version table for the dependency set |
| `tests/test_5_baseline.py` | 3339 | Forward + CLIP loss + backward on a dummy batch |
| `tests/test_6_eval.py` | 2641 | Synthetic retrieval metric check — **reimplements** the metrics |
| `pyproject.toml` | 931 | Deps + PyTorch cu121 index config |
| `uv.lock` | 754860 | Resolved lockfile, `revision = 3`, `requires-python = ">=3.10"` |
| `.python-version` | 6 | `3.10` |
| `.gitignore` | 241 | Ignores `.venv/`, `__pycache__/`, `checkpoints/`, `data/`, `wandb/`, `*.pth`, `*.ckpt`, `.vscode/`, `.idea/` |
| `README.md` | 0 | **empty** |
| `AFRNet_Windows_Baseline_Guide.docx` | 14647 | Windows setup guide (readable; text extracted during audit) |
| `models/biomedclip/*` | ~786 MB | Local HF snapshot of the BiomedCLIP repo (§4) |

**[V]** **`data/` does not exist.** Neither as a dataset directory nor as a
Python package. `ls` of the project root and a full `find` confirm it.

**[V]** `checkpoints/` does not exist either; `BaselineTrainer.__init__`
creates it (`baseline_trainer.py:83`).

---

## 3. End-to-end execution flow (as written)

**[V]** Traced through `train.py`:

1. `parse_args()` — `--config` (default `configs/baseline.yaml`), `--smoke`, `--eval-only` (`train.py:21-26`).
2. Config existence check then `yaml.safe_load` (`train.py:32-37`).
3. `--smoke` overrides: `epochs=1`, `batch_size=8`, `num_workers=0`, `data.max_samples=200` (`train.py:39-45`).
4. Device: `torch.cuda.is_available()` → `cuda`, else prints a CPU warning and uses `cpu` (`train.py:48-55`).
5. `local_dir` resolved against `Path(__file__).parent` if relative (`train.py:59-63`).
6. `BiomedCLIPBaseline(local_model_dir=..., freeze_backbone=...)` (`train.py:65-70`).
7. `data_root` resolved the same way; **exits 1** if absent, telling the user to run the download script (`train.py:73-81`).
8. `from data.roco_dataset import build_roco_loaders` → `(train_loader, val_loader)`, passed `config`, `model.tokenizer`, `model.preprocess_train`, `model.preprocess_val` (`train.py:85-90`).
9. `BaselineTrainer(model, train_loader, val_loader, config, device)` (`train.py:93-94`).
10. If `--eval-only`: `trainer.load_checkpoint("best.pt")` then `RetrievalEvaluator(model, device).evaluate(val_loader)` and return (`train.py:96-101`).
11. Else `trainer.train()` (`train.py:104`).
12. Final: `load_checkpoint("best.pt")` → `RetrievalEvaluator(...).evaluate(val_loader)` → formatted print block (`train.py:107-119`).

**[V]** **Step 8 fails with `ModuleNotFoundError`.** The import is lazy
(inside `main()`), so steps 1-7 run first: the model is fully constructed —
including any network download — *before* the pipeline aborts.

**[V]** Step 7's guard also means `--eval-only` requires the dataset on disk.

**[V]** Asset-acquisition flow, `scripts/download_roco.py`:
`set_hf_env()` repoints `HF_HOME`/`HF_DATASETS_CACHE`/`HUGGINGFACE_HUB_CACHE`/`TORCH_HOME`
into `data/hf_cache` (`:43-50`) → `download_dataset()` loops
`["train","validation","test"]`, `load_dataset("eltorio/ROCOv2-radiology", split=…)`,
`ds.save_to_disk(data/roco/<split>)`, writes a `.done` marker, up to 5 retries
with 15s·n backoff (`:53-112`) → `download_model()` `snapshot_download` of the
BiomedCLIP repo into `models/biomedclip` with `local_dir_use_symlinks=False`
(`:119-159`).

**[V]** `os.environ["HF_HOME"]` is set *before* `datasets`/`huggingface_hub`
are imported in both functions, which is the order that actually works.

---

## 4. Model architecture and pretrained-weight loading

### 4.1 Declared architecture

**[V]** `models/biomedclip/open_clip_config.json` (the authority):

```
embed_dim: 512
vision_cfg: timm_model_name=vit_base_patch16_224, timm_model_pretrained=false,
            timm_pool="", timm_proj="linear", image_size=224
text_cfg:   hf_model_name      = microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
            hf_tokenizer_name  = microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
            hf_proj_type       = mlp
            hf_pooler_type     = cls_last_hidden_state_pooler
            context_length     = 256
preprocess_cfg: CLIP mean/std (0.48145466,0.4578275,0.40821073 / 0.26862954,0.26130258,0.27577711)
```

So: timm ViT-B/16 vision trunk + linear proj head; HF BERT-base text trunk +
2-layer MLP projection; shared 512-d space.

### 4.2 Actual checkpoint contents

**[V]** Inspected `models/biomedclip/open_clip_pytorch_model.bin` without
loading torch (zip + `pickletools.dis` on `test/data.pkl`):

- Format: PyTorch zip (`PK\x03\x04`), 356 zip members, 783,705,670 bytes.
- **352 state-dict keys**, **no `state_dict` wrapper key**.
- Prefix counts: `text.*` = 200, `visual.*` = 151, `logit_scale` = 1.
- Vision keys: `visual.trunk.patch_embed.proj.{weight,bias}`,
  `visual.trunk.cls_token`, `visual.trunk.pos_embed`,
  `visual.trunk.blocks.{0..11}.{attn.qkv,attn.proj,mlp.fc1,mlp.fc2,norm1,norm2}.*`,
  `visual.trunk.norm.*`, `visual.head.proj.weight`.
- Text keys: `text.transformer.embeddings.{word_embeddings,position_embeddings,token_type_embeddings,LayerNorm}.*`,
  `text.transformer.embeddings.position_ids`, 12 encoder layers under
  `text.transformer.encoder.layer.{0..11}.*`, and `text.proj.0.weight`,
  `text.proj.2.weight`.
- **Absent:** `text.token_embedding`, `text.positional_embedding`,
  `text.transformer.embeddings.token_type_ids`.

This confirms the docstring at `models/baseline.py:6-14` ("visual.trunk.*,
text.transformer.* keys", "784 MB, 352 keys") and confirms the text tower is a
**HuggingFace BERT**, not open_clip's native text transformer.

**[I]** Parameter count ≈ ViT-B/16 (~86 M) + BERT-base (~109 M) + projections
≈ **~196 M fp32 ≈ 784 MB**, consistent with the file size.

### 4.3 How `BiomedCLIPBaseline` loads

**[V]** `models/baseline.py:31` — `HF_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"`.

**[V]** Two-step strategy as written:

1. `open_clip.create_model_and_transforms(HF_MODEL_ID)` (`:70-72`) plus
   `open_clip.get_tokenizer(HF_MODEL_ID)` (`:74`).
2. If `local/open_clip_pytorch_model.bin` exists: `torch.load(..., map_location="cpu", weights_only=False)` (`:82-86`),
   unwrap an optional `"state_dict"` key (`:89-90`), `pop("text.transformer.embeddings.position_ids")` (`:92`),
   then `self.model.load_state_dict(state_dict, strict=True)` (`:94-96`).
3. `embed_dim` discovered by a dummy `encode_image(torch.zeros(1,3,224,224))` under `no_grad` (`:113-115`).

**[V]** The `.pop` of `position_ids` is **necessary**: the checkpoint contains
that key (§4.2) and a `strict=True` load would reject it as unexpected.
**[I]** It works because `transformers` ≥ 4.31 registers `BertEmbeddings.position_ids`
as a **non-persistent** buffer, so it is absent from `model.state_dict()`.
With `transformers==4.40.0` pinned this holds. A version change in either
direction can break the strict load. *Check:* build a bare BERT from config and
test whether `"position_ids" in [k for k in m.state_dict()]`.

**[V]** Because the load is `strict=True`, the `if missing:` / `if unexpected:`
branches at `:98-103` are **unreachable** — `load_state_dict` with
`strict=True` either raises `RuntimeError` or returns two empty lists. The
"WARNING missing keys" path can never print. (`AFR-008`)

**[V] (upgraded from [I] on 2026-10-18)** **The docstring's offline claim is
wrong**, confirmed by reading the installed `open_clip 3.3.0` - the same version
`uv.lock` pins, so its source is authoritative. Corrections follow the original
paragraph.

**[I] (original wording, partly superseded)** `models/baseline.py:8-9`,
`:39-43` and `:68` assert that step 1 only reads a cached 1 KB config and "does
NOT download the 784 MB weights". In open_clip, an `hf-hub:`-prefixed model
name with no `pretrained=` argument resolves the config *and* the checkpoint
from the Hub and loads it; building the text tower additionally calls
`AutoModel.from_pretrained(hf_model_name)` because `CLIPTextCfg.hf_model_pretrained`
defaults to `True` and this config does not override it — pulling
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (~440 MB), a repo that
is **not** present in `models/biomedclip/`. `get_tokenizer(HF_MODEL_ID)`
likewise constructs an `HFTokenizer` from `hf_tokenizer_name`, i.e. that same
remote repo — so the local `tokenizer.json`, `vocab.txt`,
`tokenizer_config.json` and `special_tokens_map.json` in
`models/biomedclip/` are **never read** by this code path.
Net effect: construction requires network, downloads ~1.2-1.5 GB, and the
subsequent local `load_state_dict` overwrites most of what was just fetched.

**Corrected [V], from `site-packages/open_clip/factory.py` (v3.3.0):**

- The `hf-hub:` branch calls `_get_hf_config` (downloads
  `open_clip_config.json`), then `download_pretrained_from_hf(model_id)`
  ("Found default weights file on HF Hub"), then
  `load_checkpoint(model, checkpoint_path, strict=True)`. So the **~784 MB CLIP
  checkpoint is downloaded and loaded** - the manual Step 2 is redundant.
- **But BiomedBERT weights are NOT downloaded.** Because `checkpoint_path is
  not None`, the factory sets `enable_default_text_weights = False` and hence
  `text_cfg['hf_model_pretrained'] = False`, so `HFTextEncoder` uses
  `AutoModel.from_config` (`hf_model.py`). Only
  `AutoConfig.from_pretrained(hf_model_name)` runs - a small config fetch from
  a repo still absent locally. **Revised hidden cost: ~784 MB, not 1.2-1.5 GB.**
  The "~440 MB BiomedBERT" figure in the paragraph above is **withdrawn**.
- open_clip's own `load_checkpoint` already deletes
  `text.transformer.embeddings.position_ids`, duplicating the manual pop at
  `models/baseline.py:92`.
- **`open_clip 3.3.0` supports a `local-dir:` schema** that the first audit
  missed. `get_model_config` reads `<dir>/open_clip_config.json`;
  `create_model_and_transforms` calls `_find_checkpoint_in_dir(<dir>)`, whose
  preferred-name list includes `open_clip_pytorch_model.bin`; and
  `get_tokenizer("local-dir:<dir>")` is documented in-source to ALWAYS use the
  local dir path for HFTokenizer, so the currently-dead local tokenizer files
  **would** be used. This is the recommended fix (`AFR-004`).

**[U]** Still unverified by execution: that the `local-dir:` path actually
loads, because `transformers` is not installed anywhere.
*Check:* run it with `HF_HUB_OFFLINE=1` after `uv sync`. (`AFR-004`)

**[V]** The model card itself documents the correct local recipe —
`models/biomedclip/README.md:125-172`: read `open_clip_config.json`, inject
`model_cfg` into `open_clip.factory._MODEL_CONFIGS` under a local name
(`"biomedclip_local"`), then
`create_model_and_transforms(model_name=..., pretrained="<local .bin>", **image_<preprocess_cfg>)`.
That path takes the local weights directly and needs no `hf-hub:` id.

**[V]** `models/biomedclip/.done` is a **0-byte** marker written by
`scripts/download_roco.py:148` only after `snapshot_download` returns. It
records "the snapshot call succeeded once"; it is **not** a checksum, a
manifest, or evidence that any particular file is complete or uncorrupted —
and it says nothing about the separate BiomedBERT repo that loading also needs.

**[V]** `local_model_dir` is resolved with `.resolve()` and `use_local`
requires `weights_path.exists()` (`:61-63`). If the file is missing the wrapper
prints a warning and silently proceeds with whatever step 1 produced
(`:107-110`) — i.e. **pretrained-from-hub or, if that is also unavailable, an
unusable model**. There is no hard failure. Distinguish carefully:
architecture construction ≠ pretrained initialization.

### 4.4 Public surface

**[V]** `BiomedCLIPBaseline` (an `nn.Module`):

| Member | Signature / behaviour |
|---|---|
| `encode_image(images)` | `[B,3,H,W] → [B,512]`, L2-normalized (`:125-127`) |
| `encode_text(tokens)` | `[B,L] → [B,512]`, L2-normalized (`:129-131`) |
| `tokenize(texts)` | delegates to `self.tokenizer(texts)`, CPU (`:133-135`) |
| `clip_loss(img_feats, txt_feats)` | symmetric CE (§7) (`:141-152`) |
| `get_patch_tokens(images)` | `visual.trunk.forward_features(images)[:, 1:, :]` (`:158-164`) |
| `get_word_tokens(tokens)` | **broken** (`:166-175`, see below) |
| `freeze_backbone()` / `unfreeze_backbone()` | `:181-190` |
| `trainable_params()` / `total_params()` / `param_summary()` | `:192-203` |
| attributes | `model`, `preprocess_train`, `preprocess_val`, `tokenizer`, `embed_dim` |

**[V]** `get_word_tokens` (`:172-174`) accesses `self.model.text.token_embedding`,
`self.model.text.positional_embedding`, and calls
`self.model.text.transformer(emb.permute(1,0,2))`. None of those exist on an
open_clip `HFTextEncoder`: §4.2 shows the only `text.*` parameters are
`text.transformer.*` (a BERT taking `input_ids`, batch-first) and `text.proj.*`.
This raises `AttributeError` on first use. (`AFR-009`)

**[I]** `get_patch_tokens` is plausible: `visual.trunk` is a timm ViT, whose
`forward_features` returns `[B, 1+196, 768]`, so dropping index 0 gives
`[B,196,768]` as documented. Not executed. **[U]** unconfirmed at runtime.

**[V]** `freeze_backbone` iterates `self.model.named_parameters()` and keeps
only the parameter literally named `"logit_scale"` trainable (`:182-184`).
`config.model.freeze_backbone` is `false`, so this is inactive by default.

---

## 5. Image and text preprocessing

**[V]** Preprocessing transforms are **not defined in this repository**. They
come from `open_clip.create_model_and_transforms`, which returns
`preprocess_train` and `preprocess_val` built from `preprocess_cfg`
(§4.1 mean/std, 224×224). `train.py:88-89` hands both to
`build_roco_loaders`, so the missing dataset module is responsible for
applying them.

**[I]** Standard open_clip behaviour: `preprocess_train` =
RandomResizedCrop(224, scale≈(0.9,1.0)) + RGB convert + ToTensor + Normalize;
`preprocess_val` = Resize(224) + CenterCrop(224) + RGB + ToTensor + Normalize.
**[U]** The exact composition for open_clip 3.3.0 is unverified (no venv).
*Check:* `print(model.preprocess_val)` once installed.

**[V]** `albumentations==2.0.8` is a declared dependency but is imported by no
file in the repository (`grep` across all `.py`). It is presumably intended for
the dataset module.

**[V]** Tokenization: `BiomedCLIPBaseline.tokenize` and the loader use
`model.tokenizer`. The config's `context_length` is 256.
**[V] (resolved 2026-10-18)** `open_clip 3.3.0`'s `get_tokenizer` resolves the
context length as `context_length = text_config.get('context_length',
DEFAULT_CONTEXT_LENGTH)`, i.e. it **reads 256 from the model config
automatically**. Callers do **not** need to pass `context_length=256`; the
explicit argument in `models/biomedclip/README.md:101-104` and
`tests/test_3_biomedclip.py` is redundant but harmless. Verified by reading
`site-packages/open_clip/factory.py` (installed version == locked version).
**[U]** The resulting tensor shape has not been observed, since building a
tokenizer needs `transformers`.

---

## 6. Dataset schema and pairing assumptions

**[V]** Dataset identifier used by code: `eltorio/ROCOv2-radiology`
(`scripts/download_roco.py:33`). Splits `train`, `validation`, `test` (`:35`).

**[V]** `scripts/download_roco.py:9-14` claims ~60k train / ~9.9k validation /
~9.9k test and `save_to_disk` format. `:106-110` reads `sample.get("caption","")`,
implying a `caption` column, and prints `list(sample.keys())` — i.e. the
author did not hard-code the schema.

**[U]** **The real ROCOv2 schema is unverified.** Nothing has been downloaded;
no cached copy exists. Column names (`image`? `caption`? `cui`?), the image
encoding, and the exact per-split counts are unknown.
*Check:* after download, `datasets.load_from_disk("data/roco/validation").features`.

**[V]** `AFRNet_Windows_Baseline_Guide.docx` §7 names the dataset
`eltorio/ROCO` — a **different identifier** from the script's
`eltorio/ROCOv2-radiology`. (`AFR-014`)

**[U]** **Pairing validity is unverified and is the key scientific risk.** The
evaluator assumes one positive per row on the diagonal
(`evaluators/retrieval_eval.py:90`). Radiology caption corpora commonly contain
duplicate or near-duplicate captions and multiple images per study, which would
make several off-diagonal entries true positives and **understate** R@K and
MRR. *Check:* after download, count duplicate captions and duplicate image
hashes per split before trusting any metric. (`AFR-011`)

**[V]** `data.max_samples` (`configs/baseline.yaml:23`, overridden to 200 by
`--smoke`) is consumed only by the missing loader. No current file reads it.

---

## 7. Training objective and tensor dimensions

**[V]** `models/baseline.py:141-152`:

```
logit_scale = model.logit_scale.exp().clamp(max=100.0)     # scalar
logits_i2t  = logit_scale * img_feats @ txt_feats.T        # [B,B]
labels      = arange(B)                                    # [B]
loss        = (CE(logits_i2t, labels) + CE(logits_i2t.T, labels)) / 2
```

**[V]** Symmetric InfoNCE / CLIP loss. Inputs are `[B,512]` and already
L2-normalized by `encode_*`, so `logits` are scaled cosine similarities.
`logit_scale.exp().clamp(max=100.0)` reproduces open_clip's standard clamp.
`clip_loss` reuses `logits_i2t.T` rather than recomputing — correct and cheaper.

**[V]** Expected loss at chance is `ln(B)`; the guide's "loss ~1.39" for
`tests/test_5_baseline.py` is `ln(4)`, consistent.

**[V]** Shapes end to end: `[B,3,224,224]` images and `[B,256]` tokens →
`[B,512]` + `[B,512]` → `[B,B]` logits → scalar loss.

**[V]** `logit_scale` participates in the same AdamW parameter group as
everything else despite the comment at `baseline_trainer.py:89`
("separate LR for logit_scale") — `:90-96` builds a single flat group. The
comment is inaccurate; behaviour is uniform LR and uniform weight decay,
including on `logit_scale`, biases and LayerNorm weights (CLIP implementations
usually exclude those from decay).

---

## 8. Optimizer, scheduler, mixed precision, gradient clipping

**[V]** `trainers/baseline_trainer.py:90-96` — `AdamW(lr=config.training.lr,
weight_decay=0.1, betas=(0.9,0.98), eps=1e-6)` over
`[p for p in model.parameters() if p.requires_grad]`.

**[V]** `:99-103` — `total_steps = epochs * len(train_loader)`,
`warmup_steps = warmup_epochs * len(train_loader)`;
`cosine_schedule_with_warmup` (`:37-51`) is a `LambdaLR`: linear ramp
`step/warmup_steps` then `0.5*(1+cos(pi*progress))` floored at
`min_lr_ratio=0.0`. `scheduler.step()` is called **per batch** (`:216`),
matching the step-based horizon. At `current_step == 0` the multiplier is
exactly `0.0`, so the first optimizer step has lr 0 (`AFR-018`, cosmetic).

**[V]** `:86-87` — `amp_dtype = bfloat16 if torch.cuda.is_bf16_supported() else float16`;
`GradScaler(enabled=(amp_dtype == torch.float16))`, imported from the
deprecated `torch.cuda.amp` namespace (`:27`).

**[V]** `:194` — the autocast context is **hard-coded** to
`device_type="cuda"`, unconditionally.

**CORRECTED 2026-10-18.** The first audit claimed CPU execution "is not
supported" and crashes in the constructor. **Measured on the system
`torch 2.13.0+cpu`, it does not crash:**

| Call | Observed |
|---|---|
| `torch.cuda.is_available()` | `False` |
| `torch.cuda.is_bf16_supported()` | **`False`** - no exception; its source begins `if not is_available(): return False` |
| `torch.autocast(device_type="cuda", ...)` on CPU | `UserWarning: CUDA is not available ... Disabling autocast.`, then runs |
| `torch.cuda.amp.GradScaler(enabled=True)` | `FutureWarning` (deprecated) **+** `UserWarning: ... CUDA is not available. Disabling.`; `is_enabled()` -> `False` |
| full fp16 AMP step (scale -> unscale_ -> clip -> step) | **SUCCEEDED**, loss dtype `float32`, gradients finite |

So on torch 2.13 the CPU path **degrades silently to fp32** rather than
failing. The residual defect is that the degradation is silent and unintended:
the banner at `baseline_trainer.py:132` still prints
`AMP dtype : torch.float16`, which is misleading. Severity was lowered from
High to Medium. **[U]** Behaviour on the **pinned** torch 2.5.1 is unverified -
it may lack the early `is_available()` guard.
*Check:* `CUDA_VISIBLE_DEVICES="" uv run python -c "import torch; print(torch.cuda.is_bf16_supported())"`.
(`AFR-003`)

**[V]** Precision vs. real hardware: the GPU is **sm_75 (Turing)**. bf16 has no
native tensor-core support before sm_80. `:85` comments "bfloat16 is native on
RTX 5090 (Blackwell)" — hardware that is not present.
**[V]** The source of `is_bf16_supported` (torch 2.13) shows the decision
path: it returns `True` immediately when `get_device_properties(device).major
>= 8`, and otherwise - which is the sm_75 case - falls through to
`_check_bf16_tensor_supported(device)`, an **emulation** probe, unless
`including_emulation=False`. The trainer calls it with the default
`including_emulation=True`. So on this GPU it may well return `True` via
emulation and select bf16 on hardware with no native bf16 tensor cores.
**[U]** Not observable here (system torch is CPU-only; pinned torch is 2.5.1).
Run the check above on the GTX 1650 under the project venv before trusting
either branch. (`AFR-005`)

**[V]** Gradient clipping (`:203-205`, `:211-213`) uses `max_norm=self.grad_clip`
over `self.model.parameters()` (all params, not just trainable). The fp16 path
correctly orders `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)`
→ `clip_grad_norm_` → `scaler.step()` → `scaler.update()`. The bf16 path does
plain `backward()` → clip → `step()`, with no scaler, which is correct for bf16.

**[V]** `total_loss += loss.item()` per batch (`:217`) forces a host sync each
step — a throughput cost, not a correctness issue.

**[V]** No gradient accumulation, no `torch.compile`, no channels-last, no
distributed support. `zero_grad(set_to_none=True)` is used (`:191`).

**[V]** **No random seed is set anywhere** in the repository
(`grep -rn "manual_seed|random.seed|np.random.seed"` → no hits). Runs are not
reproducible. (`AFR-010`)

---

## 9. Evaluation metrics — definitions and implementation

**[V]** `evaluators/retrieval_eval.py`. `RetrievalEvaluator(model, device, ks=(1,5,10))`.

**`encode_dataset(loader)`** (`:43-67`): `model.eval()` under `@torch.no_grad()`,
iterates `(images, tokens)`, calls `encode_image`/`encode_text`, appends
`.cpu()` tensors, returns `img_feats [N,512]`, `txt_feats [N,512]`.
**[V]** No autocast here — evaluation runs in fp32 while training ran in
bf16/fp16. Defensible, but note the train/eval precision difference.
**[V]** `model.eval()` is set but **never restored to `train()`**. The trainer
calls `self.model.train()` at the top of each `_train_epoch` (`:181`), so the
loop recovers. Any future caller that evaluates mid-epoch would not.

**`compute_metrics(img_feats, txt_feats)`** (`:69-121`):

```
compute_dev = cuda if self.device.type == "cuda" else cpu
sim = img_feats @ txt_feats.T            # [N,N], row = image, col = text
gt  = arange(N)
for k in ks:
    k_eff = min(k, N)
    R@k i2t = mean( gt_i in topk(sim[i])     )
    R@k t2i = mean( gt_i in topk(sim.T[i])   )
rank_i2t = argmax(argsort(sim,   desc)[i] == i) + 1
rank_t2i = argmax(argsort(sim.T, desc)[i] == i) + 1
MRR      = mean(1/rank);   median_rank = median(rank)
```

**[V] VERIFIED BY EXECUTION 2026-10-18.** `tools/check_evaluator.py` imports
this production class and asserts hand-computed values; **11/11 passed, exit
0**. Perfect alignment (N=8 orthonormal) gave R@1 = MRR = 1.0 and median rank
1. A controlled swap of captions 0 and 1 produced ranks exactly `[2,2,1,1]`,
hence R@1 = 0.5, R@5 = 1.0, MRR = 0.75 - all matched. `evaluate()` through the
2-tuple loader contract returned all 10 keys. So the formulas are correct;
`AFR-006` concerns memory only.

Two behaviours were pinned down that reading alone had missed:

- **`torch.median` takes the LOWER middle value for even N.** On ranks
  `[1,1,2,2]` it returns `1.0`, while `torch.quantile(..., 0.5)` returns `1.5`.
  So `*_median_rank` is not the statistical median for even N. This surfaced
  because an assertion failed and the **expectation**, not the code, was wrong.
- **Strict diagonal scoring demonstrably understates recall.** Two identical
  images paired to two identical captions produced `i2t_R@1 = 0.7500` where the
  grouped-positive truth is `1.0000` - a correct retrieval counted as an error.
  This makes the `AFR-011` mechanism verified; only its magnitude on real
  ROCOv2 is unknown.

**[V]** Checked against the standard definitions:

- **Ranking direction** is correct: `descending=True`, `topk` returns largest similarities.
- **Ground-truth indexing** is the diagonal, consistent between R@K and MRR.
- **i2t vs t2i orientation** is correct: `sim` rows are images, `sim.T` rows are texts.
- **R@K** is `mean(any(topk == i))` — correct.
- **MRR** is `mean(1/rank)` with 1-indexed ranks — correct.
- **Median rank** is `median(rank)` as a float — correct, and the key name says `median_rank`, which is what it computes.
- **`k_eff = min(k, N)` [V]**: confirmed by execution. At `N=3`, both
  `i2t_R@5` and `i2t_R@10` exist and both hold R@3. Metric names therefore
  misrepresent the quantity on small splits (e.g. a smoke run). Low severity,
  real.
- **Empty loader [V]**: observed - `evaluate([])` raises
  `ValueError: torch.cat(): expected a non-empty list of Tensors` from
  `encode_dataset`, before `compute_metrics`. Unhandled, but it fails loudly.
- **Ties** are broken by `topk`/`argsort` index order, not explicitly. With
  duplicate captions this flatters low indices arbitrarily. Standard practice,
  worth knowing.
- **Device placement**: features are gathered on CPU then moved to
  `compute_dev`; `gt` is created on `compute_dev`. Consistent.

**[V]** **Memory is the real defect.** `compute_metrics` materialises, on the
compute device: `sim` `[N,N]` fp32, then `argsort` `[N,N]` **int64**, then a
bool `[N,N]`, then `.float()` `[N,N]` fp32 — twice over (i2t and t2i), with
`sorted_i2t` still live when the t2i block runs. At the documented validation
size `N ≈ 9,900`: `sim` ≈ 392 MB, each `argsort` ≈ 784 MB, each bool ≈ 98 MB,
each `.float()` ≈ 392 MB → **peak well above 3 GB**, on a 4 GB card that also
holds a ~196 M-parameter model. OOM is the expected outcome. (`AFR-006`)

**[V]** A rank computation that needs no sort and no `[N,N]` int64:
`rank = (sim > sim.diag().unsqueeze(1)).sum(dim=1) + 1`. It equals the
`argsort`-based rank when similarities within a row are distinct; under ties it
yields the optimistic rank rather than `argsort`'s index-order rank. Any such
change alters the protocol and must be recorded in
`docs/EXPERIMENT_LOG.md` before baseline numbers are compared.

**`evaluate(loader)`** (`:123-142`) = `encode_dataset` → `compute_metrics`
→ pretty print → return dict. Keys produced:
`i2t_R@{1,5,10}`, `t2i_R@{1,5,10}`, `i2t_MRR`, `t2i_MRR`,
`i2t_median_rank`, `t2i_median_rank` — 10 metrics, matching the project brief.

**[V]** **Validity caveat that outranks all of the above:** the diagonal
assumption is unverified for ROCOv2 (§6). If duplicate captions exist, every
number here is a lower bound of unknown slack.

---

## 10. Checkpoint format and restoration behaviour

**[V]** `trainers/baseline_trainer.py:226-237` — `torch.save` of a dict:

```
{"epoch": int, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(),
 "metrics": dict, "best_r1": float}
```

**[V]** `model` is the `BiomedCLIPBaseline` wrapper, so keys carry a `model.`
prefix (`model.visual.trunk.*`) — **not** interchangeable with the raw
open_clip state dict in `models/biomedclip/`. This is an invariant.

**[V]** Written to `checkpoint_dir / filename`, `checkpoint_dir` from
`config.checkpoint_dir` (default `"checkpoints"`), `mkdir(parents=True,
exist_ok=True)` in `__init__` (`:82-83`). **[V]** `checkpoint_dir` is
**relative to the process CWD**, unlike the model and data paths which
`train.py` anchors to the project root — so running from another directory
scatters checkpoints. (`AFR-017`)

**[V]** Two files: `best.pt` on improved `i2t_R@1` (`:158-162`) and `latest.pt`
every epoch (`:164`). `latest.pt` is saved with `metrics={}` (`:164`), so it
carries no evaluation record.

**[I]** Each file is ≈ 784 MB of weights + ≈ 1.57 GB of AdamW state
(`exp_avg` + `exp_avg_sq`, fp32) ≈ **2.3 GB**; two files ≈ 4.7 GB, with
`latest.pt` rewritten every epoch. `checkpoints/` is gitignored
(`.gitignore:16`). (`AFR-016`)

**[V]** `load_checkpoint(filename="best.pt")` (`:239-248`) raises
`FileNotFoundError` if absent, then `torch.load(path, map_location=self.device)`
and restores `model_state`, `optim_state`, `best_r1`.
**[V]** It does **not** restore `epoch`, the **scheduler**, the **GradScaler**,
or any **RNG state**; `train.py` has no `--resume` flag and never calls
`load_checkpoint` before `train()`. Therefore the format supports **model
restoration only, not genuine training resumption**. (`AFR-010`)

**[V]** `torch.load` is called without `weights_only=True` in both the trainer
(`:243`) and the model wrapper (`models/baseline.py:85`, explicitly
`weights_only=False`). For self-produced files this is a convenience; it means
the loader will execute arbitrary pickle payloads, so never point it at an
untrusted `.pt`.

**[V]** `--eval-only` (`train.py:96-101`) constructs the full `BaselineTrainer`
first — which creates `checkpoints/`, builds an AdamW optimizer and a
scheduler, and touches CUDA AMP state — then loads `best.pt` including
`optim_state`. Functional but wasteful, and it inherits `AFR-003` on CPU.

---

## 11. Configuration reference

**[V]** `configs/baseline.yaml`, with the consumer of each key:

| Key | Value | Read by |
|---|---|---|
| `model.name` | `biomedclip_baseline` | **nothing** (`AFR-017`) |
| `model.local_dir` | `models/biomedclip` | `train.py:59` → `BiomedCLIPBaseline` |
| `model.freeze_backbone` | `false` | `train.py:69` |
| `data.root` | `data/roco` | `train.py:73-81` (existence guard) |
| `data.max_samples` | `null` (→200 under `--smoke`) | missing loader only |
| `training.epochs` | `10` | `baseline_trainer.py:76` |
| `training.batch_size` | `128` | printed at `:133`; applied by the missing loader |
| `training.lr` | `1.0e-5` | `:77` |
| `training.weight_decay` | `0.1` | `:78` |
| `training.grad_clip` | `1.0` | `:79` |
| `training.warmup_epochs` | `1` | `:80` |
| `training.eval_every_epochs` | `1` | `:81` |
| `training.num_workers` | `8` | missing loader only |
| `checkpoint_dir` | `checkpoints` | `:82` |
| `wandb.enabled` | `false` | `:109` |
| `wandb.project` | `afrnet-baseline` | `:114` |

**[V]** `batch_size` and `num_workers` are **not applied by any existing
file** — the DataLoader that would use them does not exist. `train.py`'s
`--smoke` override of `num_workers` is likewise inert today.

**[V]** Stale comments: "Tuned for RTX 5090 (32 GB VRAM)" and "32 GB VRAM
handles 128 at bf16" (`:5`, `:25`) describe absent hardware;
`"If this folder is missing, training will fall back to HuggingFace remote"`
(`:17`) understates §4.3 — the remote is contacted either way.

---

## 12. Dependency and hardware constraints

**[V]** `pyproject.toml`: `requires-python = ">=3.10"`; pinned
`torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`
from an explicit `pytorch-cu121` index via `[tool.uv.sources]`;
`transformers==4.40.0` pinned exactly; `open-clip-torch>=3.3.0`,
`timm>=1.0.26`, `datasets>=4.8.5`, `albumentations>=2.0.8`, `pot>=0.9.6.post1`,
`einops`, `scipy`, `scikit-learn`, `pandas`, `pillow`, `tqdm`, `pyyaml`, `wandb`.

**[V]** Resolved in `uv.lock` (`revision = 3`): torch `2.5.1+cu121`,
open-clip-torch `3.3.0`, transformers `4.40.0`, timm `1.0.27`, datasets
**`5.0.0`**, huggingface-hub `0.36.2`, albumentations `2.0.8`, pillow `12.3.0`,
wandb `0.28.0`, numpy `2.2.6` (py3.10) / `2.4.6` (3.11) / `2.5.0` (≥3.12),
opencv-python-headless `5.0.0.93` (pulled in by albumentations).

**[V]** `open-clip-torch 3.3.0`'s declared dependencies are `ftfy`,
`huggingface-hub`, `regex`, `safetensors`, `timm`, `torch`, `torchvision`,
`tqdm` — **`transformers` is not among them**. It is needed only for HF text
towers, which this model has, so the `transformers==4.40.0` pin is the user's
and is load-bearing.

**[V]** Version-divergence risks, none resolved:

- The model card's tested combination is `open_clip_torch==2.23.0` +
  `transformers==4.35.2` (`models/biomedclip/README.md:55`). The project runs
  `3.3.0` + `4.40.0`.
- `transformers 4.40.0` (April 2024) predates NumPy 2.0 (June 2024), yet the
  lock resolves `numpy>=2.2.6`. **[U]** Unverified whether 4.40.0 imports
  cleanly under NumPy 2.x. *Check:* `uv run python -c "import transformers"`.
- `pyproject.toml` says `datasets>=4.8.5`; the lock picked `5.0.0`. **[U]**
  Unverified whether the `save_to_disk`/`load_from_disk` semantics used by
  `scripts/download_roco.py` are unchanged in 5.0.0.
- `pytest` is **not a dependency** anywhere, consistent with `tests/` being
  plain scripts (`AFR-012`).

**[V]** Hardware, measured: `nvidia-smi` → **NVIDIA GeForce GTX 1650, 4096 MiB,
driver 592.82, compute capability 7.5**.
`torch 2.5.1+cu121` ships sm_75 kernels, so the CUDA/driver stack is
**appropriate for this GPU** — the cu121 pin is fine. What is wrong is every
assumption built on 32 GB and Blackwell:

- `batch_size: 128` at 224² with a ~196 M-parameter model fine-tuned end to end
  will not fit in 4 GB. **[I]** A workable starting point is 8-16 with
  gradient accumulation; this needs measurement, not a guess.
- bf16 has no native support on sm_75 (§8).
- `tests/test_2_cuda.py:32-38` will print the `⚠ WARN`/`❌ WARN` branch
  (4.0 GB < 16 GB), which is correct behaviour for this machine.

**[V]** Environment state: **no `.venv`**, no HF cache at
`~/.cache/huggingface/hub`, no `data/`, no `checkpoints/`. `uv 0.13.0` is
installed. The only registered interpreter is **Python 3.13.6**, so
`uv sync` must first fetch Python 3.10 per `.python-version`. (`AFR-013`)

---

## 13. Commands and checks actually performed

Executed during this audit (read-only):

```
git status / git log --oneline -5
ls -la ; find . -path ./.git -prune -o -type f -printf "%10s %p\n"
cat .gitignore .python-version main.py pyproject.toml train.py configs/baseline.yaml
cat -n models/baseline.py trainers/baseline_trainer.py evaluators/retrieval_eval.py
cat -n scripts/download_roco.py ; cat tests/*.py
cat models/biomedclip/open_clip_config.json models/biomedclip/tokenizer_config.json
grep -n -i "BiomedNLP|open_clip|context_length" models/biomedclip/README.md
sed -n '125,180p' models/biomedclip/README.md
unzip AFRNet_Windows_Baseline_Guide.docx ; word/document.xml -> text (full guide read)
uv --version ; python --version ; py -0p ; ls ~/.cache/huggingface/hub
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
awk over uv.lock for resolved versions and open-clip-torch dependency edges
python -I: zipfile + pickletools.dis on models/biomedclip/open_clip_pytorch_model.bin
python -I -m compileall -q models trainers evaluators scripts tests train.py main.py
```

**Result:** `compileall` exited **0** — all project modules are syntactically
valid under Python 3.13. This proves syntax only.

### Second pass, 2026-10-18 (execution-based)

The system interpreter has `torch 2.13.0+cpu` and `open_clip 3.3.0` (but not
`transformers`, `datasets` or `pytest`), so these ran without installing
anything and without network access:

```
python -X utf8 tools/check_imports.py       -> exit 1; flags train.py:84 -> data.roco_dataset
python -X utf8 tools/check_evaluator.py     -> exit 0; 11/11 numeric checks passed
python -I       tests/test_1_python.py      -> UnicodeEncodeError (cp1252) on a pipe, exit 1  [AFR-020]
python -X utf8 -I tests/test_1_python.py    -> PASS branch, exit 0
  forced-FAIL copy (minor >= 99)            -> printed FAIL, ALSO exit 0                      [AFR-012]
torch.cuda.is_available()                   -> False      (system torch is CPU-only)
torch.cuda.is_bf16_supported()              -> False      (no exception)
torch.autocast(device_type="cuda") on CPU   -> UserWarning, then runs
torch.cuda.amp.GradScaler(enabled=True)     -> FutureWarning + auto-disabled
full fp16 AMP step on CPU                   -> SUCCEEDED in fp32, grads finite               [AFR-003]
read site-packages/open_clip/{factory,model,hf_model}.py  (v3.3.0 == locked)                 [AFR-004]
```

**Not performed** (require approval — install, network, or compute):
`uv sync`; any `import torch`; `tests/test_1..6`; `scripts/download_roco.py`;
`train.py --smoke`; any training; any metric measurement; any check of the
ROCOv2 schema, the open_clip 3.3.0 source, `is_bf16_supported()` on this GPU,
or peak memory.

---

## 14. Known defects and technical risks

Full register with evidence, reproduction and proposed fixes:
[docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md). Summary:

| ID | Sev | Issue |
|---|---|---|
| AFR-001 | **Blocker** | `data/roco_dataset.py` missing; pipeline cannot run |
| AFR-002 | **Blocker** | `.gitignore` ignores `data/`, the loader's own package path |
| AFR-003 | Med | CPU path silently disables AMP (corrected from High/crashes) |
| AFR-004 | High | Not offline; local tokenizer unused; ~784 MB hidden download; `local-dir:` fixes it |
| AFR-005 | High | Config assumes 32 GB RTX 5090; actual GPU is 4 GB sm_75 |
| AFR-006 | High | Evaluator O(N²) argsort → OOM at N≈9.9k (arithmetic VERIFIED correct) |
| AFR-007 | High | No test split; final results reported on the model-selection split |
| AFR-008 | Med | `strict=True` makes the missing/unexpected-key warnings dead code |
| AFR-009 | Med | `get_word_tokens()` targets a non-existent native text tower |
| AFR-010 | Med | No resume support, no seeds — runs not reproducible |
| AFR-011 | Med | Diagonal scoring understates recall (mechanism VERIFIED; rate unmeasured) |
| AFR-012 | Med | Tests assert nothing and exit 0 even on FAIL (VERIFIED); `tools/` added |
| AFR-013 | Med | No `.venv`; only Python 3.13.6 present vs `.python-version` 3.10 |
| AFR-014 | Low | `eltorio/ROCO` (guide) vs `eltorio/ROCOv2-radiology` (script) |
| AFR-015 | Low | Version divergence from the model card; `position_ids` pop is version-coupled |
| AFR-016 | Low | ~2.3 GB per checkpoint; `latest.pt` saved with empty metrics |
| AFR-017 | Low | Unused `model.name`; `checkpoint_dir` is CWD-relative |
| AFR-018 | Low | lr = 0 on step 0; deprecated `torch.cuda.amp.GradScaler` import |
| AFR-019 | Low | Dummy CPU forward in `__init__` just to read `embed_dim` (512 is in the config) |
| AFR-020 | Med | Emoji `print()` raises `UnicodeEncodeError` on redirected stdout (Windows) |

---

## 15. Unresolved questions requiring the user's input

1. **ROAM-Med / AFR-Net specification.** No equations, shapes, losses or paper
   exist in the repo (§1). Required before any research module is written.
2. **Hardware intent.** Is the GTX 1650 the training machine, or is a 32 GB GPU
   coming? This decides whether to re-tune the config for 4 GB (batch 8-16 +
   accumulation + fp16) or leave it as-is.
3. **Evaluation protocol.** Hold out `test` for the final number and use
   `validation` only for model selection? That is the standard fix for
   `AFR-007` and requires `build_roco_loaders` to return three loaders.
4. **Multi-positive handling.** If ROCOv2 contains duplicate captions
   (`AFR-011`), should the evaluator credit any matching caption (grouped
   ground truth) or keep strict diagonal matching? This choice must be fixed
   *before* a baseline is recorded, since results are not comparable across
   protocols.
5. **Dataset identity.** ROCOv2-radiology (script) or ROCO (guide)? (`AFR-014`)
6. **`data/` naming.** Rename the Python package (e.g. `afr_data/`) or
   un-ignore a tracked `data/` subtree? (`AFR-002`)
7. **Offline requirement.** Must the model load with no network at all? If so,
   switch to the `local-dir:` schema (§4.3) and vendor the BiomedBERT
   `config.json` locally (`AFR-004`).

**Resolved since the first pass** (no longer open questions): whether
`get_tokenizer` applies `context_length=256` automatically (**yes**, §5);
whether the retrieval formulas are correct (**yes**, verified by execution, §9);
whether the CPU path crashes (**no** on torch 2.13, §8); and the true size of
the hidden download (**~784 MB**, not 1.2-1.5 GB, §4.3).
