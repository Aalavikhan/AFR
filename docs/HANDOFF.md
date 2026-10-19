# HANDOFF — AFR-Net → segmentor session

**Audience:** the Claude Code session working in the repository that has the
**segmentor**. You are being handed a complete-but-unexecuted AFR-Net
implementation and asked to plug the segmentor in and get training running.

**Written:** 2026-10-20, against branch `afrnet-modules`, commit `a56ebc7`.

---

## 0. Read this first — three facts that change how you should work

1. **Nothing in `models/modules/`, `models/afrnet.py` or
   `tests/test_afrnet_modules.py` has ever been executed.** Not one forward
   pass. It was written away from the research machine. Every shape, gradient
   and numerical claim is an **inference from reading the code**, not an
   observation. Your first job is to run the tests, not to trust them.
2. **Training cannot start yet.** Three blockers, listed in §7. The segmentor
   is *not* one of them — its hooks are already built and wired.
3. **Do not modify the segmentor.** It is external by design. It enters
   AFR-Net through exactly two optional keyword arguments and nowhere else.
   Do not import it from `models/`; a test enforces that
   (`test_no_module_references_a_segmentor`).

Confirm you are on the right branch before anything else:

```bash
git branch --show-current     # must be afrnet-modules
git log --oneline -1          # must be a56ebc7
```

---

## 1. Mandatory reading order

Read these before changing a line. They are the authority; this handoff is a
map, not a replacement.

| Order | File | Why |
|---|---|---|
| 1 | `CLAUDE.md` | operating rules, environment facts, working protocol (§9) |
| 2 | `docs/FIDELITY_AUDIT.md` | **the most important file.** Per-operation traceability, every adaptation, and all open WARN items |
| 3 | `docs/SEGMENTOR_INTEGRATION.md` | the integration contract in full |
| 4 | `models/afrnet.py` | the composed forward pass |
| 5 | `docs/MODULE_SPEC.md` | what each module is intended to do |
| 6 | `docs/DOVE_SPEC.md` | the paper's equations, transcribed verbatim |
| 7 | `docs/KNOWN_ISSUES.md` | AFR-001 … AFR-020, the pre-existing defect register |
| 8 | `models/modules/*.py` | the implementations; docstrings cite equation numbers |
| 9 | `docs/VERIFICATION.md` | what has and has not been verified, and how |

The DOVE paper PDF is **gitignored** (copyright), so it is not in this
checkout. `docs/DOVE_SPEC.md` contains all 31 equations verbatim; you do not
need the PDF.

---

## 2. What the model is

Medical image–text retrieval on ROCOv2-radiology. A **frozen** BiomedCLIP
backbone (ViT-B/16 + PubMedBERT) with four trainable modules on top:

| Module | File | Role | Origin |
|---|---|---|---|
| **GTGA** | `models/modules/gtga.py` | text: local clinical phrase structure | **custom** (depthwise conv) on DOVE DTGA's Eq 13–16 skeleton |
| **DDTP** | `models/modules/ddtp.py` | vision: text-conditioned selection, 196 → 98 patches | **custom**, no DOVE counterpart |
| **ROAM-Med** | `models/modules/roam_med.py` | cross-modal fusion (IFA) + text guidance (IGA) | DOVE Eq 17–28, **inputs adapted** |
| **OT-Align** | `models/modules/ot_align.py` | auxiliary optimal-transport loss | **custom**, no DOVE counterpart |

**Never describe DDTP or OT-Align as DOVE modules.** DOVE contains no patch
selection, no optimal transport and no depthwise convolution. Misattributing
them would be a research-integrity error.

Only the four modules train (~a few M parameters). The backbone's ~196 M
parameters are frozen, `logit_scale` included, via
`BiomedCLIPBaseline.freeze_all()`.

---

## 3. The forward pass, with exact shapes

`AFRNet.forward(images, tokens, patch_mask=None, region_prior_external=None)`

```
images [B,3,224,224]                     tokens [B,L]  (L <= 256)
        |                                        |
        | backbone.get_patch_tokens              | backbone.encode_word_tokens
        | (frozen, under no_grad)                | (frozen, under no_grad)
        v                                        v
patches [B,196,768]                      words [B,L,768] + text_mask [B,L]
        |                                        |
        |                                        v  GTGA
        |                                  f_g [B,L,768]   (padding exactly 0)
        |                                        |
        |                                        v  masked mean
        |                                  text_query [B,768]
        |                                        |
        +---------> DDTP(patches, text_query, patch_mask=...)   <-- HOOK 1
        |                 |
        |                 v
        |           f_r [B,98,768], indices [B,98] (ascending),
        |           scores [B,98], mask [B,98]
        |
        v  SpatialPyramidPool(scales=(1,2,3,7))
  f_m [B,63,768]
        |
        +--> ROAM-Med
        |      IFA(f_m, f_r, region_prior=<-- HOOK 2, region_mask=sel.mask)
        |          Eq 19: S_MR = sigmoid(F'_M F'_R^T) * prior   [B,63,98]
        |          Eq 20/21/22                     -> f_mr [B,161,768]
        |      IGA(f_r, f_g, text_mask)
        |          Eq 26: S_RG = sigmoid(F'_R F'_G^T)  [B,B]  <-- batch-coupled
        |          Eq 27/28                        -> f_rg [B,768]
        v
  v_final = head(mean(f_mr))   [B,E]        t_final = head(f_rg)      [B,E]
  v_global = head(mean(f_m))   [B,E]        t_global = head(text_query) [B,E]
  ot_loss  = OTAlign(f_r, f_g, masks)       scalar
```

Returns `AFRNetOutput(v_final, t_final, v_global, t_global, ot_loss, aux)`.
`aux` is a dict with every intermediate — `patches`, `f_g`, `f_m`, `f_r`,
`f_mr`, `f_rg`, `s_mr`, `s_rg`, `ddtp_indices`, `ddtp_scores`, `ddtp_mask`,
`ddtp_all_scores`, `region_prior`, `ot`. Use it for diagnostics (§8).

`E` = `embed_dim` from config; `null` keeps 768 with identity heads.

**[INFERENCE, not observed]** 196 patches and 768 dims come from
`models/biomedclip/open_clip_config.json` (`image_size: 224`,
`timm_model_name: vit_base_patch16_224`) and from inspecting the checkpoint's
keys. 224/16 = 14, 14² = 196. **Verify at runtime before trusting it** (§6).

---

## 4. Where the segmentor plugs in — precisely

Two optional keyword arguments on `AFRNet.forward`. Both default to `None`,
and the model runs fully without them
(`test_afrnet_runs_without_any_segmentation_input`).

### HOOK 1 — `patch_mask`: hard eligibility

| Property | Requirement |
|---|---|
| Shape | `[B, 196]` |
| Dtype | `bool` (anything where `!= 0` means valid is accepted) |
| Semantics | `True` = patch **may** be selected; `False` = excluded |
| Flows to | `DDTP.forward(..., patch_mask=...)` → `models/modules/ddtp.py` |
| Effect | masked patches get `-inf` relevance and **can never be selected** |
| Also affects | OT-Align, **only** if `ot_align.source: patches` in the config |

Degenerate case: if fewer than 98 patches are valid, the surplus slots are
marked `mask=False` and zeroed rather than passed downstream as if they were
real regions (`test_ddtp_flags_slots_when_valid_patches_are_too_few`). The
effective `Nr` then varies per sample, which changes IFA/IGA statistics
batch to batch — be aware when interpreting results.

### HOOK 2 — `region_prior_external`: soft emphasis

| Property | Requirement |
|---|---|
| Shape | `[B, 196]` |
| Dtype | float |
| Range | **`[0, 1]` expected** — treated as a probability, not logits |
| Semantics | larger = more diagnostically relevant |
| Flows to | gathered at the selected indices (`prior = region_prior_external.gather(1, sel.indices)` → `[B, 98]`), then into ROAM-Med's IFA at **Eq 19** |
| Effect | `S_MR ← sigmoid(F'_M F'_R^T) ⊙ prior`, broadcast over the `Nm` axis |
| Normalisation | with `roam.prior_normalise: true`, rescaled to mean 1 over valid regions, so it reweights without changing magnitude |

**Critical:** when you pass `region_prior_external`, `AFRNet` sets
`prior_is_logits=False`, so your values are `clamp_min(0)`-ed and used
directly — **not** passed through a sigmoid. If you hand it raw logits they
will be clipped at 0 and interpreted as probabilities. Convert to `[0,1]`
yourself. When the hook is unused, DDTP's own scores are used instead with
`prior_is_logits=True` (they are `cosine/tau`, genuinely logits).

### Pixel mask → patch space

`avg_pool2d(mask, kernel_size=16)` on a `[B,1,224,224]` float mask gives each
patch the **fraction of its area** covered, then flatten row-major to
`[B,196]`. That is the natural soft prior; threshold it for `patch_mask`.
Full recipe in `docs/SEGMENTOR_INTEGRATION.md` §4.

### Gradients

`patch_mask` is boolean → no gradient. `region_prior_external` is float → if
it arrives with `requires_grad=True`, **gradient will flow back into the
segmentor.** Detach it unless joint training is intended; joint training
contradicts "the segmentor stays separate", so treat that as a user decision,
not yours.

---

## 5. ⚠ The trap that will silently corrupt training

**Preprocessing alignment.** BiomedCLIP's `preprocess_train` is
**RandomResizedCrop** — a different random crop and resize every epoch. If you
segment the *original* image and pool that mask, the mask describes a
different field of view from the tensor the ViT actually saw. The
misalignment is silent, plausible-looking, and poisons training.

Two ways out:

1. Apply the identical geometric transform to the mask, or
2. **Run the segmentor on the already-preprocessed 224×224 tensor** — much
   simpler, aligned by construction. Recommended. Costs segmentation quality
   at low resolution, buys correctness.

Whichever you choose, state it explicitly in `docs/EXPERIMENT_LOG.md`.

---

## 6. Verify the coordinate convention before you trust any mask

The `[B,196]` layout is assumed to be **row-major over a 14×14 grid**: index
`i` is row `i // 14`, column `i % 14`, matching `get_patch_tokens` after the
CLS token at index 0 is dropped. **This has never been observed at runtime.**
A transposed or flipped mask fails silently.

Test it directly: build a `patch_mask` that is `True` only on the left half
(`i % 14 < 7`), run a forward pass, and assert every returned
`aux["ddtp_indices"]` satisfies `i % 14 < 7`. Also confirm
`get_patch_tokens(x).shape == (B, 196, 768)`. Do this **before** wiring the
real segmentor.

Also note: 224×224 is coarse for radiology. Each patch covers ~140×180 pixels
of a native chest X-ray, so thin structures (vessels, fracture lines, small
nodules) may occupy under 25 % of every patch they cross and be thresholded
away. For small or elongated pathology prefer the **soft prior** over the
hard `patch_mask`.

---

## 7. Blockers before training can start

In the order they must be resolved.

### Blocker 1 — `data/roco_dataset.py` does not exist (AFR-001)

`train.py:84` runs `from data.roco_dataset import build_roco_loaders` and
there is no `data/` package. Nothing runs end to end.

You must implement `build_roco_loaders(config, tokenizer, preprocess_train,
preprocess_val)`. **Hard contract** (`CLAUDE.md` §6):

- yield **`(images, tokens)` 2-tuples**: `[B,3,224,224]` float and `[B,L]`
  int64 — every consumer iterates exactly this
- apply `preprocess_train` / `preprocess_val`
- honour `data.max_samples`, `training.batch_size`, `training.num_workers`
- **never shuffle validation** — the evaluator's ground truth is the diagonal
- image `i` must pair with caption `i`

Note `AFR-002`: `.gitignore` ignores `data/`, which is also this module's
import path, so the file would be untracked. Either rename the package (e.g.
`afr_data/`, updating `train.py:84`) or narrow the ignore. Ask the user which.

To feed the segmentor you will likely need the loader to emit a third element
(the mask or the raw image). That changes the 2-tuple contract, so add it as
an **opt-in** path and leave the baseline contract intact.

### Blocker 2 — the trainer does not train AFRNet

`trainers/baseline_trainer.py` still trains `BiomedCLIPBaseline` only. It
needs to: build the optimizer from `model.trainable_parameters()` (**not**
`model.parameters()`), call `AFRNet.forward`, and use
`models/modules/losses.py::AFRNetObjective`.

Inherited defects to fix or work around while you are in there — all in
`docs/KNOWN_ISSUES.md`: `AFR-003` (hard-coded CUDA autocast), `AFR-010` (no
seed, no resume), `AFR-016` (checkpoint size), `AFR-020` (emoji prints crash
on redirected stdout — always invoke with `python -X utf8`).

### Blocker 3 — `WARN-EVAL-1`: retrieval is `O(N²)`

**This blocks producing any metric, and it is a user decision, not yours.**

`v_final` and `t_final` are **cross-conditioned**: DDTP conditions the visual
side on the text query, and IGA conditions the text side on the regions.
Neither embedding exists without the other. So
`evaluators/retrieval_eval.py`, which calls `encode_image` and `encode_text`
*independently*, cannot be used — `AFRNet.encode_image/encode_text`
deliberately raise `NotImplementedError` rather than return something
silently unconditioned.

Retrieval would need N² forward passes: ~98 M at N ≈ 9,900. Not feasible.
Options: (a) retrieve-then-rerank with cheap unconditioned embeddings for the
shortlist; (b) make the visual side text-independent, keeping conditioning on
the text side only, which also matches DOVE's actual asymmetry; (c) evaluate
on a small subset and report its size. **Ask the user to choose.**

---

## 8. Two diagnostics to run early

Both are cheap and both test concerns that would otherwise show up as an
unexplained null result.

1. **Is IFA doing anything?** Log `aux["s_mr"]`'s mean and variance after a
   few hundred steps. `f_m` and `f_r` are both derived from the same 196 patch
   tokens (unlike DOVE, where they came from two independent encoders), so
   `S_MR` may saturate toward a constant and the gate carry no information.
   Low variance = IFA has degenerated. See `FIDELITY_AUDIT.md` §3.1.
2. **Magnitude balance in Eq 20/21.** Eq 19 is a sigmoid with **no**
   normalisation and no `1/√d`, so the attended term's magnitude scales with
   the token count. DOVE had `Nm=4, Nr=36`; this has `Nm=63, Nr=98`, making
   the Eq 21 attended term up to an order of magnitude larger relative to its
   residual. Try `roam.pyramid_scales: [1, 2]` (`Nm=5`) to get closer to the
   paper's balance. Compare the norms of the attended term and the residual.

---

## 9. Decisions that belong to the user, not to you

Do not pick these silently. Each changes what the results mean.

| Decision | Where | Default now |
|---|---|---|
| Retrieval protocol (`WARN-EVAL-1`) | §7 Blocker 3 | unresolved — blocks all metrics |
| `roam.iga_mode` — `batch` is faithful to Eq 26 but couples the mini-batch, so a sample's embedding depends on its batch-mates; `diagonal` removes that but is an algorithmic modification | `configs/afrnet.yaml` | `batch` |
| `gtga.branch_mode` — `directional` restores DTGA's forward/backward asymmetry; `symmetric` mines kernel width instead and may degenerate | `configs/afrnet.yaml` | `symmetric` (**`directional` recommended**, with equal kernels e.g. `[5,5]`) |
| `loss.lambda_ot` (0.1, untuned) and `lambda_g` (0; the paper's 10.0 was **not** transplanted) | `configs/afrnet.yaml` | as shown |
| `ddtp.ratio` — 0.5 keeps 98/196; likely too generous when pathology is 1–5 % of pixels | `configs/afrnet.yaml` | 0.5 |
| Joint training of the segmentor | §4 Gradients | off (detach) |
| `data/` package rename (`AFR-002`) | §7 Blocker 1 | unresolved |

---

## 10. Rules you must not break

From `CLAUDE.md` §9 — read it in full, but these matter most here:

1. **Never claim code works because it reads correctly.** Only executed output
   counts. Nothing here has run.
2. **Label every claim** `VERIFIED` / `INFERENCE` / `UNKNOWN`.
3. **Eq 19 and Eq 26 use sigmoid, not softmax**, with no `1/√d`. Do not
   "modernise" them into normalised attention — that replaces the module.
   Guarded by `test_eq19_is_sigmoid_not_softmax`.
4. **Do not modify the segmentor**, and do not import it from `models/`.
5. **Never fabricate masks.** If segmentation is unavailable, pass `None`.
6. **Do not silently change** the data splits, the retrieval protocol, the
   loss weights or the primary objective. The primary loss is deliberately
   the **existing** InfoNCE, not DOVE's triplet loss, so the baseline stays
   comparable.
7. **Never tune on the test split.** `AFR-007`: no test split is wired in at
   all yet, and the final figure is currently measured on the
   model-selection split. Fix that before recording any baseline.
8. **Ask before** `uv sync`, downloading weights or datasets, or any training
   beyond a smoke run.
9. **Record every real run** in `docs/EXPERIMENT_LOG.md`. No run, no entry.

---

## 11. Suggested order of work

1. Read §1's file list. Confirm the branch and commit.
2. `python -X utf8 tests/test_afrnet_modules.py` — **expect failures.** Fix
   them; the likeliest are the causal/anti-causal padding indices in
   `test_gtga_directional_*`, the Sinkhorn marginal tolerances, and shapes in
   the `StubBackbone` integration tests.
3. `python -X utf8 tools/check_imports.py` — exits 1 until Blocker 1 is done.
4. With the user's approval, `uv sync`, then verify the backbone contract:
   `get_patch_tokens` → `[B,196,768]` and `encode_word_tokens` → `[B,L,768]`
   plus a mask. This is also the first real test of the `AFR-009` fix.
5. Verify the patch coordinate convention (§6) with a synthetic half-image
   mask, **before** wiring the real segmentor.
6. Implement Blocker 1 (the dataset loader), masks optional.
7. Get a `--smoke` forward+backward pass running **without** segmentation.
   Confirm the backbone receives no gradients and all four modules do.
8. Add the segmentor via the two hooks. Run the §8 diagnostics.
9. Resolve Blocker 3 with the user, then and only then produce metrics.
10. Record everything in `docs/EXPERIMENT_LOG.md`.

Get a clean forward pass with **no** segmentation first. If you add the
segmentor at the same time and results shift, you will not know which change
caused it.
