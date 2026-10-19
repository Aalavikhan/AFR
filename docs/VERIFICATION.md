# AFR-Net — Verification Inventory, Gaps and Plan

What this repository can currently prove about itself, what it cannot, and the
cheapest path to closing the gap.

Compiled 2026-10-18 at commit `7858ee5`. Tags: **[V]** verified by execution ·
**[I]** inference · **[U]** unverified.

---

## 1. Inventory of existing validation tools

| Tool | Kind | Asserts? | Exit code on failure | Needs |
|---|---|---|---|---|
| `tests/test_1_python.py` | print script | no | **0** | stdlib |
| `tests/test_2_cuda.py` | print script | no | **0** | torch + CUDA |
| `tests/test_3_biomedclip.py` | print script | no | **0** | torch, open_clip, transformers, **~784 MB download** |
| `tests/test_4_deps.py` | print script | no | **0** | the full dependency set |
| `tests/test_5_baseline.py` | print script | no | **0** | as test_3, **~784 MB download** |
| `tests/test_6_eval.py` | print script | no | **0** | torch |
| `tools/check_imports.py` | real check | yes | **1** | stdlib |
| `tools/check_evaluator.py` | real check | yes | **1** | torch (CPU) |

`tools/*` were added on 2026-10-18 by the audit. Everything under `tests/` came
from `AFRNet_Windows_Baseline_Guide.docx` §5-6 as **manual setup checks**, which
is what they are good at — they are not a regression suite.

There is **no** `pytest`, `conftest.py`, `pytest.ini`, `.pre-commit-config.yaml`,
CI workflow or `.github/` directory anywhere in the repository. **[V]** (`find`,
`ls`.)

---

## 2. What was actually proven on 2026-10-18

All of the following was executed. Commands and output are in §6.

**[V] The retrieval metric arithmetic is correct.** `tools/check_evaluator.py`
imports the production `RetrievalEvaluator` and asserts 11 hand-computed values:
perfect alignment gives R@1 = MRR = 1.0 and median rank 1; a controlled
two-caption swap gives ranks `[2,2,1,1]`, so R@1 = 0.5, R@5 = 1.0 and
MRR = 0.75 exactly. **11/11 passed, exit 0.** This retires the suspicion that
`AFR-006` was about wrong formulas — it is purely a memory problem.

**[V] `torch.median` uses the lower-middle convention.** On ranks `[1,1,2,2]`
it returns `1.0`, while `torch.quantile(..., 0.5)` returns `1.5`. So
`i2t_median_rank` / `t2i_median_rank` report the lower of the two central ranks
for even `N`, not their average. Not a bug; a reporting convention that must be
stated when the metric is published. This was found *because* an assertion
failed and the **expectation**, not the code, turned out to be wrong.

**[V] Strict diagonal ground truth really does understate recall.** With two
identical images paired to two identical captions, retrieving caption 0 for
image 1 is correct in reality, but the diagonal demands column 1 and the tie
breaks to column 0 — the evaluator reports `i2t_R@1 = 0.75` where the grouped
truth is `1.0`. The *mechanism* behind `AFR-011` is now demonstrated.
**[U]** The real ROCOv2 duplicate rate remains unmeasured, so the *magnitude*
is still unknown.

**[V] `R@K` is mislabelled when `N < k`.** At `N = 3`, `i2t_R@5` and
`i2t_R@10` both exist and both hold R@3, because of `k_eff = min(k, N)` at
`evaluators/retrieval_eval.py:95`. Matters for smoke runs.

**[V] An empty loader fails loudly.** `evaluate([])` raises
`ValueError: torch.cat(): expected a non-empty list of Tensors` from
`encode_dataset`, before `compute_metrics`.

**[V] A failing test exits 0.** `tests/test_1_python.py` exits 0 on its PASS
branch, and a forced-FAIL copy (`minor >= 99`) printed
`❌ FAIL — Need Python 3.10+, got 3.13` and **also exited 0**. Confirms
`AFR-012`: no automation can detect a regression from these scripts.

**[V] `AFR-001` is reproducible automatically.** `tools/check_imports.py`
reports `LOCAL MISSING  train.py:84  ->  data.roco_dataset` and exits 1.

**[V] Emoji output crashes on redirected stdout (new — `AFR-020`).**
`python tests/test_1_python.py` with stdout as a pipe raised
`UnicodeEncodeError: 'charmap' codec can't encode character '✅'` from
`cp1252`, exit 1. Nine project files print non-ASCII, including `train.py` (3
lines) and `trainers/baseline_trainer.py` (3 lines). `-X utf8` fixes it;
`PYTHONIOENCODING` does **not** help under `-I`, because `-I` implies `-E`.

**[V] `open_clip 3.3.0` source settles `AFR-004`.** The installed version equals
the locked version, so `site-packages/open_clip/factory.py` is authoritative:
an `hf-hub:` id downloads `open_clip_config.json` *and* the default weights file
and loads it (`factory.py` hf-hub branch → `load_checkpoint`). Two corrections
to the earlier audit are recorded in `docs/KNOWN_ISSUES.md` AFR-004.

---

## 3. Verification gaps, by consequence

### Gap 1 — Nothing guards the pieces that would silently corrupt results
**Severity: highest.** The components whose failure is *silent* rather than
loud have no checks at all:

- the loader's image↔caption pairing (`AFR-001`, not written yet);
- split separation, i.e. that train/val/test do not overlap (`AFR-007`);
- checkpoint round-tripping, i.e. that `best.pt` reloads to identical weights;
- determinism, i.e. that two runs with a fixed seed agree (`AFR-010`, no seed
  exists anywhere).

A bug in any of these produces plausible-looking numbers. That is far more
dangerous for a research baseline than a crash.

### Gap 2 — `tests/test_6_eval.py` tests a copy, not the product
It defines its own `evaluate_retrieval()` and never imports
`evaluators.retrieval_eval`. The two implementations already differ: the copy
uses a Python `sum(i in row.tolist() ...)` loop where production uses a
vectorised `.any(dim=1).float().mean()`, and the copy omits `median_rank`
entirely. `tools/check_evaluator.py` now covers the real thing; the duplicate
logic should be deleted rather than left to drift.

### Gap 3 — No failure signal
Every `tests/` script swallows its outcome into stdout. `tests/test_5_baseline.py`
additionally wraps everything in `except Exception` + `traceback.print_exc()`
and still exits 0, so even a crash reads as success to a caller.

### Gap 4 — The model wrapper is untestable offline
`models/baseline.py:31` hard-codes the `hf-hub:` id, so *any* test touching
`BiomedCLIPBaseline` needs ~784 MB and a network round-trip. That is why the
audit could verify the evaluator but **not** the model wrapper. Fixing
`AFR-004` (the `local-dir:` schema) would make the wrapper cheaply testable —
the verification win is a reason to prioritise it.

### Gap 5 — No memory or shape guard
`AFR-006`'s `O(N²)` int64 `argsort` is confirmed to grow as predicted, but
nothing fails before an OOM on real data. `get_patch_tokens` / `get_word_tokens`
have no shape test; `get_word_tokens` is already broken (`AFR-009`) and no check
would have caught it.

### Gap 6 — Environment drift is invisible
`transformers==4.40.0` against `numpy>=2.2.6` is **[U]** untested, and the
model card's tested pair is `open_clip 2.23.0` + `transformers 4.35.2`. Nothing
asserts that the local checkpoint still loads strictly under the installed
`transformers`, so a future bump would fail at training time rather than at
check time (`AFR-015`).

---

## 4. Recommended plan

Ordered by value per unit of effort. Nothing here is implemented; items 2-5
need `uv sync` first, which needs the user's approval.

**1. Keep `tools/` as the pre-flight gate (done).**
`python -X utf8 tools/check_imports.py && python -X utf8 tools/check_evaluator.py`.
Both return meaningful exit codes and need no project environment. Run before
any pipeline attempt.

**2. Add `pytest` as a dev dependency and a real suite.** Keep the `tests/`
scripts as the guide's manual environment checks — they serve that purpose —
and add assertion-based tests beside them:

| Test | Asserts | Closes |
|---|---|---|
| `test_eval_metrics.py` | port `tools/check_evaluator.py` to pytest; add tie and `N<k` cases | Gap 2 |
| `test_loader_contract.py` | shapes, dtypes, `max_samples`, **deterministic val order**, image `i` ↔ caption `i` | Gap 1 |
| `test_split_disjoint.py` | train/val/test share no sample id | Gap 1, `AFR-007` |
| `test_checkpoint_roundtrip.py` | save → load gives identical params and every documented key | Gap 1, `AFR-016` |
| `test_determinism.py` | 2+2 steps from a checkpoint == 4 uninterrupted steps | Gap 1, `AFR-010` |
| `test_model_shapes.py` | `embed_dim == 512`, `get_patch_tokens → [B,196,768]`, `get_word_tokens` | Gap 5, `AFR-009` |
| `test_offline_load.py` | wrapper loads under `HF_HUB_OFFLINE=1` | Gap 4, `AFR-004` |
| `test_strict_load.py` | local `.bin` loads strictly under the installed `transformers` | Gap 6, `AFR-015` |

Mark anything needing the Hub as `@pytest.mark.network` and keep the default
run offline.

**3. Measure the duplicate rate** on the real validation split before recording
any baseline, and record the number in `docs/EXPERIMENT_LOG.md`. This decides
the ground-truth protocol (`AFR-011`) and the protocol must be fixed *before*
numbers exist, since the two are not comparable.

**4. Add a seed** to `configs/baseline.yaml` and seed
`random`/`numpy`/`torch`/CUDA plus the DataLoader generator (`AFR-010`).
Without it, no experiment in this project is reproducible.

**5. Pre-flight before full training:** run the evaluator on the real
validation `N` with random weights. It either OOMs — proving `AFR-006` on this
4 GB card — or it does not, and either answer is worth knowing before a 10-epoch
run.

---

## 5. What still cannot be verified here, and why

| Unknown | Blocker | Check that would settle it |
|---|---|---|
| `is_bf16_supported()` on sm_75 | system torch is CPU-only; locked torch is 2.5.1, not 2.13 | run it under `uv sync`'d torch 2.5.1 on the GTX 1650 |
| Whether `AFR-003` crashes or degrades on torch **2.5.1** | only 2.13 observable, where it degrades with a warning | `CUDA_VISIBLE_DEVICES="" uv run python -c "import torch; torch.cuda.is_bf16_supported()"` |
| Real peak VRAM / largest workable batch | no CUDA build, no dataset | measure after `uv sync` |
| ROCOv2 schema, split sizes, duplicate rate | nothing downloaded | `load_from_disk(...).features` after download |
| `transformers 4.40.0` under NumPy 2.x | not installed | `uv run python -c "import transformers"` |
| `preprocess_train/val` exact composition | needs a built model | `print(model.preprocess_val)` |
| Whether the model wrapper loads at all | needs transformers + ~784 MB | `test_offline_load.py` after `AFR-004` |

**[U] The model wrapper, the trainer loop and the dataset path have never been
executed by anyone.** Only the evaluator has been exercised. Any statement
about end-to-end behaviour is inference until `uv sync` happens.

---

## 6. Command record

Executed 2026-10-18. Output is in the session transcript; results are
summarised in §2.

```bash
# syntax
python -X utf8 -m compileall -q models trainers evaluators scripts tests train.py main.py   # exit 0

# the two new checks
python -X utf8 tools/check_imports.py      # exit 1, flags train.py:84 -> data.roco_dataset
python -X utf8 tools/check_evaluator.py    # exit 0, 11/11 numeric checks passed

# test-script behaviour
python -I tests/test_1_python.py                     # UnicodeEncodeError on a pipe (AFR-020)
python -X utf8 -I tests/test_1_python.py             # PASS branch, exit 0
#   forced-FAIL copy (minor >= 99) printed "FAIL" and also exited 0

# torch / AMP behaviour on the system interpreter (torch 2.13.0+cpu)
torch.cuda.is_available()            -> False
torch.cuda.is_bf16_supported()       -> False   (early guard: `if not is_available(): return False`)
torch.autocast(device_type="cuda")   -> UserWarning "Disabling autocast", then runs
torch.cuda.amp.GradScaler(enabled=True) -> FutureWarning + "CUDA is not available. Disabling."
full fp16 AMP step on CPU            -> SUCCEEDED in fp32, grads finite

# open_clip 3.3.0 source (== locked version), read only
site-packages/open_clip/factory.py   : hf-hub branch, local-dir branch,
                                       _find_checkpoint_in_dir, load_checkpoint,
                                       enable_default_{image,text}_weights gate
site-packages/open_clip/model.py     : CLIPTextCfg.hf_model_pretrained = True
site-packages/open_clip/hf_model.py  : AutoConfig.from_pretrained / AutoModel.from_config
```

**Not executed** (needs approval): `uv sync`, `scripts/download_roco.py`,
`tests/test_2..6`, `train.py` in any mode, anything touching the GPU or the Hub.
