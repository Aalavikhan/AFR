# AFR-Net — Known Issues Register

Opened 2026-10-18 from a read-only static audit at commit `7858ee5`.
Reference: [PROJECT_KNOWLEDGE.md](PROJECT_KNOWLEDGE.md).

**Status values:** `OPEN` · `IN PROGRESS` · `FIXED (verified)` · `WONTFIX` · `NEEDS DECISION`.
An issue moves to `FIXED (verified)` only when the change *and* its verification
evidence are both recorded here.

**No issue below has been fixed.** No source file was modified by this audit.

**2026-10-18 follow-up.** A second pass found `torch 2.13.0+cpu` and
`open_clip 3.3.0` installed on the *system* interpreter, which allowed real
execution-based verification. That evidence **corrected two findings**
(`AFR-003` severity, `AFR-004` scope), **upgraded two** to verified
(`AFR-011` mechanism, `AFR-012`), and **added `AFR-020`**. Metric arithmetic
in `evaluators/retrieval_eval.py` is now **verified correct by execution**
(`tools/check_evaluator.py`, 11/11). Details: [VERIFICATION.md](VERIFICATION.md).

Evidence tags: **[V]** verified · **[I]** inference · **[U]** unverified.

---

## AFR-001 — Dataset module does not exist (pipeline cannot run)

- **Severity:** Blocker
- **Status:** OPEN
- **Files:** `train.py:84`, absent `data/roco_dataset.py`
- **Evidence [V]:** `train.py:84` executes
  `from data.roco_dataset import build_roco_loaders`. A full `find` of the
  working tree shows no `data/` directory, no `data/__init__.py`, and no
  `roco_dataset.py` anywhere. `git status` is clean, so nothing is merely
  uncommitted, and `git log` has only two commits.
- **Reproduce:** `uv run python train.py --smoke` (after the environment and
  dataset exist) → `ModuleNotFoundError: No module named 'data'`.
- **Actual vs expected:** Expected two DataLoaders over ROCOv2; actual is an
  import error. Because the import is lazy inside `main()`, the full model is
  constructed first (§3), so the failure costs a model load before it reports.
- **Root cause [V]:** The loader was never written.
  `AFRNet_Windows_Baseline_Guide.docx` §6 lists the files to copy into the
  project and `data\roco_dataset.py` is **not** among them; §9 lists
  "Build the SkinCAP / ROCO Dataset class and DataLoader (ask Claude for
  Segment 2)" as the next step. This is deferred work, not a lost file.
- **Proposed fix:** Implement `build_roco_loaders(config, tokenizer,
  preprocess_train, preprocess_val)` honouring the contracts in
  `CLAUDE.md` §6: yield `(images, tokens)` 2-tuples, apply
  `preprocess_train`/`preprocess_val`, respect `data.max_samples`,
  `training.batch_size`, `training.num_workers`, and **never shuffle
  validation** (the evaluator's diagonal ground truth depends on order).
  Depends on `AFR-002` (package naming) and the §6 schema question, so inspect
  the downloaded dataset's `.features` before writing it.
- **Regression tests required:** a synthetic-fixture test that the loader emits
  the declared shapes/dtypes, that `max_samples` truncates, that validation
  order is deterministic across two iterations, and that image *i* pairs with
  caption *i*.

---

## AFR-002 — `.gitignore` ignores `data/`, which is also the loader's package path

- **Severity:** Blocker (for source control, not execution)
- **Status:** NEEDS DECISION
- **Files:** `.gitignore:17`, `train.py:84`, `scripts/download_roco.py:29`
- **Evidence [V]:** `.gitignore` line 17 is `data/`. `scripts/download_roco.py:29`
  writes the dataset to `ROOT/data/roco`, so ignoring it is correct for the
  data. But `train.py:84` imports the Python module `data.roco_dataset`, i.e.
  **source code** that must live at `data/roco_dataset.py` — inside the ignored
  tree. The guide's §3.2 compounds this by creating `data` with
  `mkdir data, configs, models, ...`, as a plain data directory.
- **Reproduce:** create `data/roco_dataset.py`, then `git status` → untracked
  and hidden; `git add data/roco_dataset.py` → refused without `-f`.
- **Actual vs expected:** Expected the dataset loader to be version-controlled;
  actual is that it would be silently untracked and lost on a fresh clone.
- **Root cause [V]:** One path serves two purposes — downloaded data and
  importable source.
- **Proposed fix (user's choice, see PROJECT_KNOWLEDGE §15.6):** either
  (a) move the loader to a new tracked package, e.g. `afr_data/roco_dataset.py`,
  and update `train.py:84` — cleanest, no `.gitignore` games; or
  (b) keep `data/` as a package and narrow the ignore to `data/roco/`,
  `data/hf_cache/`, with `data/*.py` tracked.
  Recommend (a): it removes the ambiguity permanently.
- **Regression tests required:** none beyond `AFR-001`'s; add a CI/pre-commit
  check that the loader module is tracked by git.

---

## AFR-003 — CPU path silently disables AMP; crash risk is version-dependent

- **Severity:** ~~High~~ → **Medium** (corrected 2026-10-18)
- **Status:** OPEN
- **CORRECTION:** the original entry claimed CPU execution "is impossible" and
  crashes in the constructor. **That was wrong for the torch actually tested.**
  Measured on `torch 2.13.0+cpu`:
  `torch.cuda.is_bf16_supported()` → `False` (its source has an early
  `if not is_available(): return False` guard, so it does **not** raise);
  `torch.autocast(device_type="cuda", ...)` on CPU emits
  `UserWarning: CUDA is not available or torch_xla is imported. Disabling autocast.`
  and runs anyway; `GradScaler(enabled=True)` self-disables with
  `UserWarning: ... CUDA is not available. Disabling.`; and a **full fp16 AMP
  training step on CPU SUCCEEDED** in fp32 with finite gradients.
  So on torch 2.13 the CPU path **degrades silently to fp32 rather than
  failing**. The remaining defect is that it is silent and unintended, plus the
  unverified behaviour on the **pinned** torch 2.5.1, which may lack that guard.
  **[U]** *Check:* `CUDA_VISIBLE_DEVICES="" uv run python -c "import torch; print(torch.cuda.is_bf16_supported())"`
  once the project venv exists.
- **Files:** `trainers/baseline_trainer.py:86`, `:194`; `train.py:48-50`
- **Evidence [V]:** `train.py:48-50` prints
  `"⚠ CUDA not available — running on CPU (very slow)"` and sets
  `device = torch.device("cpu")`. But `BaselineTrainer.__init__:86` calls
  `torch.cuda.is_bf16_supported()` unconditionally, and `_train_epoch:194`
  opens `torch.autocast(device_type="cuda", dtype=self.amp_dtype)` with the
  device type hard-coded.
- **Reproduce:** on a machine without CUDA (or with
  `CUDA_VISIBLE_DEVICES=""`), construct `BaselineTrainer`.
- **Actual vs expected:** Expected either a working CPU run or a clear error.
  Actual (torch 2.13) is a working run that **silently ignores the configured
  AMP dtype** and emits three warnings per process. A user reading
  `AMP dtype : torch.float16` in the banner at `baseline_trainer.py:132` would
  wrongly believe mixed precision was active.
- **Root cause [V]:** the device type is hard-coded rather than derived from
  `self.device`. On torch 2.13 this is absorbed by torch's own guards; whether
  torch 2.5.1 absorbs it identically is **[U]** unverified.
- **Proposed fix:** derive a single `self.device_type = self.device.type`;
  select `amp_dtype` only when `device_type == "cuda"` (else disable AMP);
  pass `device_type=self.device_type` and `enabled=(self.device_type=="cuda")`
  to `torch.autocast`. Keep the fp16/bf16 branch logic otherwise unchanged.
- **Regression tests required:** a CPU-only test that builds the trainer with a
  tiny synthetic loader and runs one `_train_epoch` step to completion.

---

## AFR-004 — "Offline local load" is not offline; local tokenizer files are never used

- **Severity:** High
- **Status:** OPEN
- **CORRECTION (2026-10-18):** the original entry estimated "~1.2-1.5 GB" and
  said BiomedBERT **weights** are pulled. Reading the **installed
  `open_clip 3.3.0`** (identical to the locked version, so authoritative)
  shows that is **partly wrong**:
  - **Confirmed [V]:** the `hf-hub:` branch of `factory.py` calls
    `_get_hf_config` (downloads `open_clip_config.json`) and then
    `download_pretrained_from_hf(model_id)` — "Found default weights file on HF
    Hub" — and `load_checkpoint(model, checkpoint_path, strict=True)` loads it.
    So the **~784 MB CLIP checkpoint is downloaded and loaded**, making the
    manual Step 2 in `models/baseline.py` redundant.
  - **Corrected [V]:** because `checkpoint_path is not None`, `factory.py` sets
    `enable_default_text_weights = False` and therefore
    `text_cfg['hf_model_pretrained'] = False`, so `HFTextEncoder` takes
    `AutoModel.from_config` — **BiomedBERT weights are NOT downloaded.**
    Only `AutoConfig.from_pretrained(hf_model_name)` runs (`hf_model.py`), a
    small config fetch from a repo still absent locally. Revised hidden cost:
    **~784 MB, not 1.2-1.5 GB.**
  - **Also [V]:** open_clip's own `load_checkpoint` already deletes
    `text.transformer.embeddings.position_ids` ("Certain text transformers no
    longer expect position_ids after transformers==4.31"), so the manual pop at
    `models/baseline.py:92` duplicates library behaviour.
- **Files:** `models/baseline.py:8-9, 31, 39-46, 68-74`; `configs/baseline.yaml:17`
- **Evidence:**
  - **[V]** `models/baseline.py:31` uses the `hf-hub:`-prefixed id, and `:72`/`:74`
    pass it to `create_model_and_transforms` and `get_tokenizer` with no
    `pretrained=` and no `cache_dir=`.
  - **[V]** The docstring at `:8-9` and `:39-43` and the print at `:69` claim
    only a sub-1 KB cached config is read and that the 784 MB weights are not
    re-downloaded. `configs/baseline.yaml:17` similarly implies the remote is a
    fallback only.
  - **[V]** `models/biomedclip/open_clip_config.json` sets
    `text_cfg.hf_model_name` and `hf_tokenizer_name` to
    `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` — a **different HF
    repo that is not present locally** (the local dir holds only the
    BiomedCLIP snapshot).
  - **[V]** There is **no HF cache** on this machine
    (`~/.cache/huggingface/hub` does not exist), so nothing is pre-cached.
  - **[I]** open_clip resolves an `hf-hub:` id by fetching both
    `open_clip_config.json` and the checkpoint from the Hub, and builds the HF
    text tower with `AutoModel.from_pretrained(hf_model_name)` because
    `CLIPTextCfg.hf_model_pretrained` defaults to `True` and this config does
    not override it. `get_tokenizer` likewise builds `HFTokenizer` from
    `hf_tokenizer_name`. Net: ~1.2-1.5 GB fetched, then step 2 overwrites most
    of it, and the local `tokenizer.json` / `vocab.txt` /
    `tokenizer_config.json` / `special_tokens_map.json` are dead files.
  - **[V]** The model card ships the correct recipe:
    `models/biomedclip/README.md:125-172` registers the local `model_cfg` in
    `open_clip.factory._MODEL_CONFIGS` under a local name and passes
    `pretrained="<local .bin>"`.
- **Reproduce:** `HF_HUB_OFFLINE=1 uv run python -c "from models.baseline import BiomedCLIPBaseline; BiomedCLIPBaseline('models/biomedclip')"`
  with an empty HF cache → expect a hub-resolution failure, not a successful
  local load.
- **Actual vs expected:** Expected a network-free load from
  `models/biomedclip/`; actual is a mandatory hub round-trip with a large
  redundant download, and a silent dependency on a second repo.
- **Root cause [V]:** architecture construction was conflated with pretrained
  initialization; the `hf-hub:` id carries both.
- **Proposed fix — simpler than the model-card recipe [V]:** `open_clip 3.3.0`
  supports a **`local-dir:` schema** that the original audit missed.
  `parse_model_name` returns `'local-dir'`; `create_model_and_transforms`
  then reads `<dir>/open_clip_config.json`, merges `preprocess_cfg`, and calls
  `_find_checkpoint_in_dir(dir)` — whose preferred-name list **includes
  `open_clip_pytorch_model.bin`**, exactly the file present. And
  `get_tokenizer("local-dir:<dir>")` comments *"If config came from local-dir,
  ALWAYS use the local dir path for HFTokenizer"*, so the **currently-dead
  local `tokenizer.json` / `vocab.txt` would actually be used.**
  `context_length` is also read from the config automatically
  (`context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)`
  → **256**), which additionally resolves the open question in
  PROJECT_KNOWLEDGE §5.
  So the change is roughly: build the id as `f"local-dir:{local_dir}"` instead
  of the `hf-hub:` constant, drop the now-redundant manual Step 2, and **fail
  loudly** instead of silently falling back (`:107-110`). The only remaining
  Hub dependency is `AutoConfig.from_pretrained(hf_model_name)` for the BERT
  architecture config; vendoring that config locally and repointing
  `hf_model_name` would make it fully offline.
  **[U] Not implemented and not executed** — it needs `transformers`, which is
  not installed. Verify with `HF_HUB_OFFLINE=1` before trusting it.
- **Regression tests required:** a test that loading succeeds with
  `HF_HUB_OFFLINE=1`, and one asserting the local tokenizer files are the ones
  actually used.

---

## AFR-005 — Configuration is tuned for hardware that is not present

- **Severity:** High
- **Status:** NEEDS DECISION
- **Files:** `configs/baseline.yaml:5, 25`; `trainers/baseline_trainer.py:7, 85-86`
- **Evidence [V]:** `nvidia-smi` reports
  **NVIDIA GeForce GTX 1650, 4096 MiB, driver 592.82, compute capability 7.5**.
  `configs/baseline.yaml:5` says "Tuned for RTX 5090 (32 GB VRAM)" and `:31`
  says "32 GB VRAM handles 128 at bf16" with `batch_size: 128`.
  `baseline_trainer.py:85` says "bfloat16 is native on RTX 5090 (Blackwell)".
- **Reproduce:** `uv run python tests\test_2_cuda.py` → will report 4.0 GB and
  take the WARN branch (`tests/test_2_cuda.py:32-38`).
- **Actual vs expected:** Expected 32 GB and sm_120; actual 4 GB and sm_75.
  Consequences: (a) **[I]** `batch_size: 128` at 224² fine-tuning a ~196 M
  parameter model cannot fit in 4 GB — OOM; (b) bf16 has no native tensor-core
  support below sm_80, so the dtype choice at `:86` is at best emulated;
  (c) full fine-tuning of both towers may be impractical here regardless of
  batch size.
- **Root cause [V]:** The config and comments were authored for a different
  machine than the one in use.
- **Proposed fix:** needs the user's decision (PROJECT_KNOWLEDGE §15.2). If the
  GTX 1650 is the training machine: measure the largest batch that fits
  (start at 8), add gradient accumulation to preserve an effective batch of
  128 — contrastive learning is sensitive to the number of in-batch negatives,
  so a batch of 8 is **not** a comparable baseline — consider
  `freeze_backbone` or partial freezing, and correct the stale comments.
- **Regression tests required:** record measured peak VRAM for the chosen batch
  size in `EXPERIMENT_LOG.md`. **[U]** No memory measurement has been taken.

---

## AFR-006 — Evaluator allocates O(N²) int64 sorts; OOM expected on the real val split

- **Severity:** High
- **Status:** OPEN
- **Files:** `evaluators/retrieval_eval.py:84-119`
- **Evidence [V]:** `compute_metrics` moves features to `compute_dev` (CUDA when
  available, `:84-86`), forms `sim` `[N,N]` (`:89`), then for ranks calls
  `sim.argsort(dim=1)` (`:109`) and `sim.T.argsort(dim=1)` (`:113`) — each an
  `[N,N]` **int64** tensor — each followed by a bool `[N,N]` comparison and a
  `.float()` `[N,N]` cast (`:110`, `:114`). `sorted_i2t` is still referenced
  when the t2i block allocates.
- **Reproduce:** evaluate the ROCOv2 validation split
  (`scripts/download_roco.py:12` documents ~9.9k samples) on the GTX 1650.
- **Actual vs expected:** Expected metrics; **[I]** actual is
  `torch.cuda.OutOfMemoryError`. At `N = 9,900`: `sim` ≈ 392 MB, each `argsort`
  ≈ 784 MB, each bool ≈ 98 MB, each `.float()` ≈ 392 MB — peak well above 3 GB
  on a 4 GB card that also holds a ~196 M-parameter model.
- **Root cause [V]:** A full sort is used where only the rank of one known
  element per row is needed.
- **Scope clarified 2026-10-18 [V]:** the **arithmetic is correct.**
  `tools/check_evaluator.py` asserts 11 hand-computed values against the
  production evaluator (perfect alignment → R@1 = MRR = 1.0; a controlled
  caption swap → ranks `[2,2,1,1]`, R@1 = 0.5, R@5 = 1.0, MRR = 0.75) and
  **all 11 passed, exit 0**. Growth was also confirmed by running N = 500 and
  N = 1500 through `compute_metrics`. This issue is therefore **purely about
  memory**, not correctness — do not "fix" the formulas.
- **Proposed fix:** replace both `argsort` blocks with
  `rank = (sim > sim.diag().unsqueeze(1)).sum(dim=1) + 1`, which needs no int64
  `[N,N]` and no sort. **Protocol note:** this equals the `argsort` rank when a
  row's similarities are distinct; under exact ties it gives the optimistic
  rank instead of `argsort`'s index-order rank. That is a protocol change and
  must be recorded in `EXPERIMENT_LOG.md`. Additionally, chunk the R@K `topk`
  over rows and offer a CPU fallback for large `N`.
- **Regression tests required:** a test on a small hand-built `sim` with known
  ranks and ties, asserting the new ranks match the old implementation's values
  on the tie-free case; plus a large-`N` smoke test of peak memory.

---

## AFR-007 — No test split; the reported baseline is measured on the model-selection split

- **Severity:** High (scientific validity)
- **Status:** NEEDS DECISION
- **Files:** `train.py:85-90, 107-119`; `trainers/baseline_trainer.py:154-162`;
  `scripts/download_roco.py:35`
- **Evidence [V]:** `scripts/download_roco.py:35` downloads
  `["train", "validation", "test"]`, so a test split **is** fetched.
  `train.py:85-90` only ever receives `train_loader, val_loader` from
  `build_roco_loaders`. `baseline_trainer.py:154-162` evaluates `val_loader`
  each epoch and saves `best.pt` on improved `i2t_R@1` — i.e. **model selection
  on validation**. `train.py:107-119` then re-evaluates the **same
  `val_loader`** and prints
  `"BASELINE RESULTS — record these carefully"`. Nothing in the repository
  reads `data/roco/test`.
- **Reproduce:** read the two call sites; the loader argument is identical.
- **Actual vs expected:** Expected a held-out test number; actual is a
  validation number optimistically biased by having selected the checkpoint
  that maximises it. The headline figure is not a clean generalization estimate,
  and the guide (§9) designates it the reference point for every later ablation.
- **Root cause [V]:** The three-way split was downloaded but never wired
  through the (missing) loader factory.
- **Proposed fix:** have `build_roco_loaders` return
  `(train_loader, val_loader, test_loader)`; keep per-epoch selection on
  `val_loader`; evaluate `best.pt` on `test_loader` for the final block in
  `train.py`; report both and label them. Then never tune against test.
- **Regression tests required:** a test asserting the three loaders draw from
  disjoint underlying splits.

---

## AFR-008 — `strict=True` makes the missing/unexpected-key warnings unreachable

- **Severity:** Medium
- **Status:** OPEN
- **Files:** `models/baseline.py:94-106`
- **Evidence [V]:** `:94-96` calls `load_state_dict(state_dict, strict=True)`.
  With `strict=True`, PyTorch either raises `RuntimeError` or returns
  `_IncompatibleKeys` with **both lists empty**. Therefore `:98-100`
  ("WARNING missing keys") and `:101-103` ("WARNING unexpected keys") can never
  execute, and `:104-106` ("Weights loaded perfectly") is the only reachable
  branch.
- **Reproduce:** inspect the call; or load a deliberately truncated state dict
  and observe a raised exception rather than the warning.
- **Actual vs expected:** The code reads as if it tolerates and reports partial
  loads; it actually hard-fails, and the diagnostics are dead.
- **Root cause [V]:** The warning block was written for `strict=False` and the
  flag was later set to `True`.
- **Proposed fix:** decide the intent explicitly. For a baseline, strict is the
  right behaviour — so delete the dead branches and keep the success print,
  wrapping the load in a `try` that reports the key-mismatch detail before
  re-raising. If partial loads must be tolerated, switch to `strict=False` and
  keep (and *test*) the warnings. Do not leave both.
- **Regression tests required:** a test that a corrupted/truncated state dict
  raises with a message naming the offending keys.

---

## AFR-009 — `get_word_tokens()` targets a text tower that does not exist

- **Severity:** Medium (blocks the planned AFR-Net work)
- **Status:** **CODE CHANGED 2026-10-18, NOT YET VERIFIED**
- **Change:** `models/baseline.py::get_word_tokens` now calls the HF BERT with
  `input_ids` + `attention_mask` and reads `last_hidden_state`;
  `pad_token_id()`, `text_pad_mask()` and `encode_word_tokens()` were added.
  The AFR-Net modules consume this interface.
- **Why this is NOT marked fixed:** per CLAUDE.md §9.5 an issue is fixed only
  once the change *and* its verification are recorded. Loading the real text
  tower needs `transformers` (not installed) and ~784 MB from the Hub, and the
  user asked that nothing be run. **Verify on the research machine with:**
  `uv run python -c "import torch;from models.baseline import BiomedCLIPBaseline as B;m=B('models/biomedclip');f,k=m.encode_word_tokens(m.tokenize(['chest x-ray']));print(f.shape,k.shape,k.sum())"`
  Expect `[1, 256, 768]`, a `[1, 256]` bool mask, and a non-zero valid count.
- **Files:** `models/baseline.py:166-175`
- **Evidence [V]:** `:172-174` use `self.model.text.token_embedding`,
  `self.model.text.positional_embedding`, and
  `self.model.text.transformer(emb.permute(1, 0, 2))` — the API of open_clip's
  *native* text transformer. Inspection of the checkpoint
  (PROJECT_KNOWLEDGE §4.2) shows the text tower is an HF BERT: the only `text.*`
  parameters are `text.transformer.embeddings.*`,
  `text.transformer.encoder.layer.{0..11}.*`, `text.proj.0.weight`,
  `text.proj.2.weight`. `text.token_embedding` and `text.positional_embedding`
  are **absent**, and BERT is batch-first, so the `permute(1,0,2)` is also wrong.
- **Reproduce:** `model.get_word_tokens(tokens)` → `AttributeError`.
- **Actual vs expected:** Expected `[B,256,768]` pre-pooling word tokens;
  actual is an attribute error on the first access.
- **Root cause [V]:** The accessor was written against the wrong text-tower
  variant.
- **Proposed fix:** call the HF module directly, e.g.
  `self.model.text.transformer(input_ids=tokens, attention_mask=(tokens != pad_id)).last_hidden_state`
  → `[B, L, 768]`. The pad id must come from the tokenizer, not be assumed.
  Verify the shape and the `hf_pooler_type` (`cls_last_hidden_state_pooler`)
  convention before any AFR-Net module consumes it.
- **Regression tests required:** a shape/dtype test for both
  `get_word_tokens` and `get_patch_tokens` (the latter's `[B,196,768]` claim is
  **[U]** unverified at runtime).

---

## AFR-010 — No training resumption and no seeding; runs are not reproducible

- **Severity:** Medium
- **Status:** OPEN
- **Files:** `trainers/baseline_trainer.py:226-248`; `train.py:21-26`
- **Evidence [V]:** `_save_checkpoint` (`:226-237`) stores only `epoch`,
  `model_state`, `optim_state`, `metrics`, `best_r1` — no scheduler state, no
  `GradScaler` state, no RNG state, no config snapshot.
  `load_checkpoint` (`:239-248`) restores `model_state`, `optim_state`,
  `best_r1` and ignores `epoch`. `train.py` exposes no `--resume` and never
  loads a checkpoint before `train()`.
  `grep -rn "manual_seed|random.seed|np.random.seed"` over the repository
  returns **no hits**.
- **Reproduce:** interrupt a run and try to continue it — the only option is to
  restart from epoch 1 with a warm model, a reset scheduler, and a different
  data order.
- **Actual vs expected:** Expected resumable, repeatable runs for a research
  baseline; actual is model restoration only, with run-to-run variation that
  cannot be separated from a method's effect.
- **Root cause [V]:** Resumption and seeding were not implemented.
- **Proposed fix:** add `seed` to `configs/baseline.yaml` and seed
  `random`/`numpy`/`torch`/CUDA plus the DataLoader `generator` and
  `worker_init_fn` in one place; extend the checkpoint with
  `scheduler_state`, `scaler_state`, RNG states and the resolved `config`; add
  `--resume <file>` to `train.py` that restores all of it and continues from
  `epoch + 1`. Keep reading old checkpoints (treat the new keys as optional) so
  any existing `best.pt` stays loadable.
- **Regression tests required:** train 2 steps, save, reload, train 2 more, and
  assert the parameters match an uninterrupted 4-step run bit for bit.

---

## AFR-011 — Diagonal ground truth may be wrong for ROCOv2 (multiple positives)

- **Severity:** Medium (potentially High — affects every reported number)
- **Status:** NEEDS DECISION
- **Files:** `evaluators/retrieval_eval.py:90, 98-119`
- **Evidence [V]:** `:90` sets `gt = torch.arange(N)` and every metric treats
  column `i` as the sole correct match for row `i`.
  **[U]** Whether ROCOv2-radiology contains duplicate or near-duplicate
  captions, or multiple images per caption, is **unverified** — nothing has
  been downloaded and no cached copy exists.
- **Reproduce:** not yet possible. *Check after download:*
  `ds = load_from_disk("data/roco/validation")`; count
  `len(ds) - len(set(ds["caption"]))` and the distribution of caption
  multiplicities; also hash images for duplicates.
- **Actual vs expected:** If duplicates exist, several off-diagonal entries are
  true positives that are scored as errors, so R@K and MRR are **lower bounds**
  with unknown slack, and the bias varies with split size and batch
  composition.
- **Mechanism DEMONSTRATED 2026-10-18 [V]:** with two identical images paired
  to two identical captions, retrieving caption 0 for image 1 is correct in
  reality, but the diagonal demands column 1 and the tie breaks to column 0 —
  the production evaluator returned `i2t_R@1 = 0.7500` where the grouped-positive
  truth is `1.0000` (`tools/check_evaluator.py` case 6). So the understatement
  is real and reproducible. **[U]** Only the *magnitude* on real ROCOv2 is
  unknown, because the dataset has still not been downloaded.
- **Root cause [V for the mechanism], [U for the rate]:** strict diagonal
  scoring plus non-unique positives. Whether ROCOv2 actually has non-unique
  positives is still unmeasured.
- **Proposed fix:** measure duplication first. If it is non-trivial, replace
  the scalar `gt` with a per-row positive **set** (group identical captions)
  and score a hit when any positive is retrieved. Fix this **before** recording
  a baseline — the two protocols are not comparable (PROJECT_KNOWLEDGE §15.4).
- **Regression tests required:** a synthetic case with two identical captions
  asserting the grouped protocol credits either match while the strict
  protocol does not.

---

## AFR-012 — Tests assert nothing, and the eval test does not test the evaluator

- **Severity:** Medium
- **Status:** OPEN
- **Files:** `tests/test_1_python.py` … `tests/test_6_eval.py`; `pyproject.toml`
- **Evidence [V]:**
  - No test file contains an `assert`, and none defines a `test_*` function —
    they are top-level scripts that `print("✅ PASS")` or `print("❌ FAIL")`.
    Exit status is `0` either way, so no CI could detect a failure.
  - `pytest` appears **nowhere** in `pyproject.toml` or `uv.lock`.
  - `tests/test_6_eval.py:16-45` defines its **own** `evaluate_retrieval()` and
    never imports `evaluators.retrieval_eval`. It therefore validates a
    private copy of the logic — passing it says nothing about the production
    evaluator. The copy also differs: it computes recall with a Python
    `sum(i in row.tolist() ...)` loop where production uses a vectorised
    `.any(dim=1).float().mean()`, and it omits `median_rank` entirely.
  - `tests/test_3_biomedclip.py` and `tests/test_5_baseline.py` both call
    `create_model_and_transforms('hf-hub:...')`, so each **downloads from the
    Hub** (and `test_5` duplicates the model wrapper's logic rather than
    importing `BiomedCLIPBaseline`).
  - `tests/test_5_baseline.py:95-98` catches every exception and prints a
    traceback, still exiting 0.
- **Reproduce:** `uv run python tests\test_6_eval.py; echo $?` → `0`
  regardless of outcome. **[U]** Not executed (no environment).
- **Actual vs expected:** Expected a suite that fails loudly on regressions;
  actual is a set of manual smoke scripts whose "PASS" is informational only.
- **Root cause [V]:** They were written as setup-verification scripts for the
  Windows guide (§5 of the guide lists them as such), not as a regression suite.
- **Partially addressed 2026-10-18:** `tools/check_evaluator.py` and
  `tools/check_imports.py` were added. Both use real assertions and **return
  meaningful exit codes** (0 / 1), and both were executed. They cover the
  evaluator and import resolution only.
- **Proposed fix (remaining):** add `pytest` as a dev dependency; keep the
  existing scripts as manual environment checks (they serve the guide) and add
  an assertion-based suite alongside. The per-test plan is in
  [VERIFICATION.md](VERIFICATION.md) §4. Mark hub-downloading tests
  `@pytest.mark.network` so the default run stays offline.
- **Regression tests required:** this issue *is* the test work; every other
  fix above lists the tests it needs.

---

## AFR-013 — Environment is not installed; interpreter mismatch

- **Severity:** Medium
- **Status:** OPEN
- **Files:** `.python-version`, `pyproject.toml`, `uv.lock`
- **Evidence [V]:** `ls -d .venv` → does not exist. `python --version` →
  **3.13.6**; `py -0p` lists **only** 3.13. `.python-version` is `3.10`;
  `pyproject.toml` requires `>=3.10`; `uv.lock` has `requires-python = ">=3.10"`
  with resolution markers covering 3.10-3.14. `uv 0.13.0` is installed.
  `~/.cache/huggingface/hub` does not exist.
- **Reproduce:** `uv run python -c "import torch"` → no environment.
- **Actual vs expected:** Expected a synced `.venv` on Python 3.10; actual is
  no environment at all, so **no runtime claim about this project has been or
  can be verified** without an install.
- **Root cause [V]:** `uv sync` has never been run in this working tree (or the
  `.venv` was removed).
- **Proposed fix:** with the user's approval, `uv python install 3.10` then
  `uv sync` (~2.5 GB of wheels). Then run `tests/test_1`-`test_4` and record
  the actual versions in `EXPERIMENT_LOG.md`, which also settles the
  **[U]** items: `transformers 4.40.0` under NumPy 2.x, `datasets 5.0.0`
  `save_to_disk` semantics, `is_bf16_supported()` on sm_75, and the open_clip
  3.3.0 download behaviour behind `AFR-004`.
- **Regression tests required:** none; this is an environment action.

---

## AFR-014 — Dataset identifier disagrees between the guide and the script

- **Severity:** Low
- **Status:** NEEDS DECISION
- **Files:** `scripts/download_roco.py:33`; `AFRNet_Windows_Baseline_Guide.docx` §7
- **Evidence [V]:** the script uses `eltorio/ROCOv2-radiology`; the guide's
  dataset table and its example command use `eltorio/ROCO`
  (`load_dataset('eltorio/ROCO', split='train')`).
- **Actual vs expected:** two different corpora (ROCO vs ROCOv2) with different
  sizes and schemas. The script is the more recent artifact and is what
  `configs/baseline.yaml` paths imply, but this has not been confirmed with the
  user.
- **Root cause [V]:** the guide predates the script.
- **Proposed fix:** confirm the intended corpus with the user, then make the
  identifier a config key (`data.hf_dataset`) instead of a module constant, so
  one place defines it.
- **Regression tests required:** none; record the chosen corpus in
  `EXPERIMENT_LOG.md`.

---

## AFR-015 — Library versions diverge from the model card's tested set

- **Severity:** Low
- **Status:** OPEN
- **Files:** `pyproject.toml`, `uv.lock`, `models/biomedclip/README.md:55`,
  `models/baseline.py:92`
- **Evidence [V]:** the model card specifies
  `open_clip_torch==2.23.0` + `transformers==4.35.2`; the project locks
  `open-clip-torch 3.3.0` + `transformers 4.40.0` with `timm 1.0.27`,
  `datasets 5.0.0`, `numpy ≥ 2.2.6`.
  **[I]** The `position_ids` pop at `models/baseline.py:92` depends on
  `transformers` registering `BertEmbeddings.position_ids` as a **non-persistent**
  buffer (true since ≈4.31). If that changes, the `strict=True` load at `:94`
  fails with a *missing* key — and per `AFR-008` it will raise rather than warn.
  **[U]** `transformers 4.40.0` under NumPy 2.x is untested here.
- **Reproduce:** `uv run python -c "import transformers, open_clip; print(transformers.__version__, open_clip.__version__)"`
  after `AFR-013`.
- **Proposed fix:** no version change yet — establish the incompatibility
  first, per the project's own rule. Add a one-line comment at
  `models/baseline.py:92` recording *why* the pop is needed and which
  `transformers` behaviour it relies on.
- **Regression tests required:** a test asserting the local checkpoint loads
  strictly under the installed `transformers`, so a future bump fails loudly.

---

## AFR-016 — Checkpoints are ~2.3 GB each and `latest.pt` records no metrics

- **Severity:** Low
- **Status:** OPEN
- **Files:** `trainers/baseline_trainer.py:164, 226-237`
- **Evidence [V]:** the saved dict contains both `model_state` and
  `optim_state`. **[I]** ~196 M fp32 parameters ≈ 784 MB plus AdamW `exp_avg` +
  `exp_avg_sq` ≈ 1.57 GB ≈ **2.3 GB per file**; `best.pt` + `latest.pt` ≈ 4.7 GB,
  with `latest.pt` rewritten every epoch (`:164`).
  **[V]** `:164` passes `metrics={}`, so `latest.pt` carries no evaluation
  record even on epochs that were evaluated.
- **Actual vs expected:** Expected modest, informative checkpoints; actual is
  heavy I/O every epoch on a research machine and a metrics field that is
  always empty in one of the two files.
- **Proposed fix:** pass the epoch's `log_dict` (or the latest `metrics`) into
  the `latest.pt` save; optionally write `optim_state` only to `latest.pt`
  (what resumption needs) and keep `best.pt` weights-only, documenting that in
  `CLAUDE.md` §6 since it changes the checkpoint contract. Note the
  interaction with `AFR-010`, which *adds* fields.
- **Regression tests required:** a test that a saved/reloaded checkpoint
  round-trips every documented key.

---

## AFR-017 — Dead config key and CWD-relative checkpoint directory

- **Severity:** Low
- **Status:** OPEN
- **Files:** `configs/baseline.yaml:15, 35`; `trainers/baseline_trainer.py:82-83`;
  `main.py`
- **Evidence [V]:** `model.name` (`baseline.yaml:15`) is read by no file
  (`grep` across the repo). `checkpoint_dir` is used as
  `Path(config.get("checkpoint_dir", "checkpoints"))` with **no anchoring to
  the project root**, unlike `model.local_dir` and `data.root`, which
  `train.py:59-63` and `:75-77` resolve against `Path(__file__).parent`.
  `main.py` is a `uv init` stub (`print("Hello from afr!")`) that nothing
  imports or calls.
- **Actual vs expected:** Running `train.py` from another directory writes
  checkpoints next to the CWD while reading the model and data from the project
  root — silently splitting the run's artifacts.
- **Proposed fix:** resolve `checkpoint_dir` against the project root the same
  way (pass an already-resolved path from `train.py`); either wire `model.name`
  into logging/run naming or delete the key; leave or remove `main.py`
  deliberately rather than by neglect.
- **Regression tests required:** a test that checkpoints land under the project
  root when `train.py` is invoked from a different CWD.

---

## AFR-018 — Cosmetic: lr is 0 on the first step; deprecated AMP import

- **Severity:** Low
- **Status:** OPEN
- **Files:** `trainers/baseline_trainer.py:27, 43-45`
- **Evidence [V]:** `lr_lambda(0)` returns `0/warmup_steps == 0.0`, so the first
  optimizer step applies zero learning rate (one wasted step out of
  `epochs * len(train_loader)`). `:27` imports `GradScaler` from
  `torch.cuda.amp`, which **[I]** emits a `FutureWarning` on torch ≥ 2.4 (the
  supported path is `torch.amp.GradScaler("cuda", ...)`); it remains functional
  in 2.5.1.
- **Proposed fix:** use `float(current_step + 1) / float(max(1, warmup_steps))`,
  and switch to `torch.amp.GradScaler`. Fold into the `AFR-003` edit, since it
  touches the same lines.
- **Regression tests required:** assert the scheduler's first lr is > 0 and
  that the warmup reaches the configured lr at `warmup_steps`.

---

## AFR-019 — Dummy CPU forward pass in `__init__` just to read `embed_dim`

- **Severity:** Low
- **Status:** OPEN
- **Files:** `models/baseline.py:113-116`
- **Evidence [V]:** `__init__` runs
  `self.model.encode_image(torch.zeros(1, 3, 224, 224))` under `no_grad` purely
  to set `self.embed_dim = feat.shape[-1]`. The value is **already declared**:
  `models/biomedclip/open_clip_config.json` → `"embed_dim": 512`. The forward
  runs on CPU (the model has not been moved to the device yet — `train.py:65-70`
  precedes `BaselineTrainer.__init__`'s `.to(device)`), and the input size 224
  is hard-coded rather than read from the config.
- **Actual vs expected:** Expected a cheap attribute; actual is a full ViT
  forward pass on CPU at every construction, including under `--eval-only`.
- **Proposed fix:** read `embed_dim` (and `image_size`) from the local
  `open_clip_config.json` — which the `AFR-004` fix already parses — and drop
  the probe, or keep it behind a `verify=True` debug flag.
- **Regression tests required:** assert `model.embed_dim == 512` without a
  forward pass.

---

## AFR-020 — Emoji in `print()` crashes on Windows when stdout is redirected

- **Severity:** Medium
- **Status:** OPEN
- **Found:** 2026-10-18 (new; not in the first audit)
- **Files:** `train.py`, `trainers/baseline_trainer.py`, `scripts/download_roco.py`,
  all six `tests/test_*.py`
- **Evidence [V]:** running `python -I tests/test_1_python.py` with stdout
  attached to a pipe produced:
  `UnicodeEncodeError: 'charmap' codec can't encode character '✅' in position 2: character maps to <undefined>`
  from `Lib/encodings/cp1252.py`, **exit 1**, after the first two lines of
  output. On Windows, Python writes to a real console via UTF-16, but falls
  back to the locale codec (**cp1252** here) when stdout is a pipe or file.
  Counted non-ASCII print lines: `scripts/download_roco.py` 12,
  `tests/test_2_cuda.py` 6, `tests/test_4_deps.py` 5,
  `tests/test_3_biomedclip.py` 4, `train.py` 3,
  `trainers/baseline_trainer.py` 3, `tests/test_5_baseline.py` 3,
  `tests/test_1_python.py` 2, `tests/test_6_eval.py` 2.
- **Reproduce:** `python tests/test_1_python.py | cat` (crashes) versus
  `python -X utf8 tests/test_1_python.py | cat` (fine).
- **Actual vs expected:** Expected a log file or piped output; actual is a
  mid-run crash. This bites exactly when it hurts most — `uv run python
  train.py > train.log`, a CI job, or `tee` during a long run. Worst case is a
  training run dying at `baseline_trainer.py:162` (`💾 New best ...`) **after**
  the epoch's compute but **before** the progress is reported.
- **Root cause [V]:** non-ASCII literals in `print()` plus Windows' non-UTF-8
  default locale encoding for redirected streams.
- **Note [V]:** `-I` implies `-E`, so `PYTHONIOENCODING=utf-8` and
  `PYTHONUTF8=1` are **ignored** when `-I` is used. `-X utf8` works.
- **Proposed fix:** cheapest is documentation plus a wrapper — always invoke
  with `-X utf8` (now in `CLAUDE.md` §8.1). A code fix would add
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at each entry
  point (`train.py`, `scripts/download_roco.py`), which is 1 line each and
  preserves the emoji. Replacing the emoji with ASCII would also work but
  touches many lines for cosmetic reasons.
- **Regression tests required:** a check that each entry point runs with stdout
  redirected under a non-UTF-8 locale.

---

## Summary

| ID | Severity | Status | One line |
|---|---|---|---|
| AFR-001 | Blocker | OPEN | `data/roco_dataset.py` missing — pipeline cannot run |
| AFR-002 | Blocker | NEEDS DECISION | `.gitignore` hides the loader's own package path |
| AFR-003 | Medium | OPEN | CPU path silently disables AMP (was "High/crashes" — corrected) |
| AFR-004 | High | OPEN | Not offline; local tokenizer unused; ~784 MB hidden download; `local-dir:` fixes it |
| AFR-005 | High | NEEDS DECISION | Config assumes 32 GB RTX 5090; GPU is 4 GB sm_75 |
| AFR-006 | High | OPEN | O(N²) int64 argsort — OOM expected (arithmetic VERIFIED correct) |
| AFR-007 | High | NEEDS DECISION | Baseline reported on the model-selection split |
| AFR-008 | Medium | OPEN | `strict=True` makes the key warnings dead code |
| AFR-009 | Medium | CODE CHANGED, UNVERIFIED | `get_word_tokens()` rewritten for the HF BERT tower |
| AFR-010 | Medium | OPEN | No resume, no seeds — not reproducible |
| AFR-011 | Medium | NEEDS DECISION | Diagonal ground truth understates recall (mechanism VERIFIED; rate unmeasured) |
| AFR-012 | Medium | OPEN (partly addressed) | Tests assert nothing and exit 0 even on FAIL (VERIFIED); `tools/` added |
| AFR-013 | Medium | OPEN | No `.venv`; only Python 3.13.6 present |
| AFR-014 | Low | NEEDS DECISION | ROCO vs ROCOv2-radiology identifier conflict |
| AFR-015 | Low | OPEN | Versions diverge from the model card's tested set |
| AFR-016 | Low | OPEN | ~2.3 GB per checkpoint; `latest.pt` metrics always empty |
| AFR-017 | Low | OPEN | Dead `model.name`; CWD-relative `checkpoint_dir` |
| AFR-018 | Low | OPEN | lr = 0 on step 0; deprecated `torch.cuda.amp` import |
| AFR-019 | Low | OPEN | CPU ViT forward in `__init__` to read a known constant |
| AFR-020 | Medium | OPEN | Emoji `print()` crashes with `UnicodeEncodeError` on redirected stdout |
