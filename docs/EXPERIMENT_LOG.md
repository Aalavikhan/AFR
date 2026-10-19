# AFR-Net — Experiment Log

Factual record of experiments actually run. One entry per run.

**Rules for this file**

- Record only outcomes that were **observed**. Never fill a field by estimate,
  expectation or analogy. Unknown fields are written `not recorded`.
- A run that crashed is still an entry — record where it failed.
- Copy metrics verbatim from the program's output, including the number of
  decimal places it printed.
- If the evaluation protocol changed (see `AFR-006`, `AFR-007`, `AFR-011` in
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md)), state which protocol the entry uses.
  Numbers from different protocols are **not comparable**.
- Note whether the figure is a **validation** number (biased by model
  selection) or a **held-out test** number.

---

## Status: no experiments have been run

As of **2026-10-18** (commit `7858ee5`), this repository has produced **no
training run, no evaluation, and no metric of any kind**.

Verified reasons:

- No Python environment exists (`ls -d .venv` → absent; only Python 3.13.6 is
  registered on the machine). — `AFR-013`
- No dataset exists (`data/` absent; nothing downloaded). — `AFR-001`
- `train.py` cannot reach its training loop: `train.py:84` imports
  `data.roco_dataset`, which does not exist. — `AFR-001`
- No checkpoints exist (`checkpoints/` absent).

Verification work *has* been done, but none of it is an experiment: it produced
no trained weights and no metric measured on real data. Executed on 2026-10-18
(full record in [VERIFICATION.md](VERIFICATION.md) §6):

- `compileall` over all modules — exit 0 (syntax only).
- `tools/check_imports.py` — exit 1, flags the missing `data.roco_dataset`.
- `tools/check_evaluator.py` — exit 0, **11/11** numeric checks against the
  production evaluator on synthetic features. This establishes that the R@K /
  MRR / median-rank arithmetic is correct; it says **nothing** about retrieval
  quality, which requires the real dataset and trained weights.

Any number produced by `tools/check_evaluator.py` is a synthetic fixture, not a
result. Do not quote it as a baseline.

**Any BiomedCLIP baseline number quoted for this project before an entry
appears below is unsourced.** The project guide
(`AFRNet_Windows_Baseline_Guide.docx` §9) designates the baseline R@1 as the
reference point for every later ablation, so it must be measured and recorded
here before any AFR-Net module is compared against it.

---

## Entry template

Copy this block for each run.

```markdown
### EXP-000 — <short name>

| Field | Value |
|---|---|
| Date (run) | YYYY-MM-DD |
| Git commit | <sha> (`git rev-parse --short HEAD`), tree clean? yes/no |
| Command | exact command line |
| Purpose | what question this run answers |
| Dataset | identifier + revision |
| Split(s) used | train / validation / test, with sample counts as printed |
| Protocol | diagonal ground truth or grouped positives; argsort rank or diag-compare rank |
| Config | path + any CLI override; paste the non-default values |
| Model init | pretrained source, or checkpoint path restored |
| Seed | value, or `not set` |
| Hardware | GPU name, VRAM, driver, compute capability (from nvidia-smi) |
| Software | python, torch, open_clip, transformers, timm, datasets versions (as printed) |
| AMP dtype | as printed by the trainer |
| Batch size / effective | configured / after gradient accumulation |
| Epochs completed | n of m |
| Wall time | as printed |
| Peak VRAM | measured, or `not recorded` |
| Status | completed / crashed / interrupted |

**Metrics** (verbatim from the program's output; state the split)

| Metric | Value |
|---|---|
| i2t_R@1 | |
| i2t_R@5 | |
| i2t_R@10 | |
| t2i_R@1 | |
| t2i_R@5 | |
| t2i_R@10 | |
| i2t_MRR | |
| t2i_MRR | |
| i2t_median_rank | |
| t2i_median_rank | |
| final train_loss | |

**Errors / warnings observed:**

**Deviations from config:**

**Reproducibility notes:** (seed set? dataloader order deterministic?
scheduler/scaler state saved? anything that would prevent an exact rerun)

**Conclusion:** (one or two sentences; no claims beyond what the numbers show)
```

---

## Pending prerequisites before the first entry

From [KNOWN_ISSUES.md](KNOWN_ISSUES.md), these must be settled or the first
recorded baseline will not be trustworthy or reusable:

1. `AFR-001` / `AFR-002` — a tracked dataset loader must exist.
2. `AFR-013` — environment installed; record the observed versions.
3. `AFR-005` — a batch size that actually fits 4 GB, with gradient accumulation
   to keep the effective batch (and therefore the negative count) comparable.
4. `AFR-006` — evaluation must survive `N ≈ 9.9k` without OOM.
5. `AFR-011` — duplicate-caption measurement, so the ground-truth protocol is
   chosen before any number is recorded.
6. `AFR-007` — final figures measured on the held-out `test` split, with
   `validation` reserved for model selection.
7. `AFR-010` — seed set, so the entry is reproducible.
