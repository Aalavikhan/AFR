# Integrating the Separate Segmentor

The segmentor is **outside** this task and was neither implemented, replaced,
retrained, refactored nor merged.

**Verified state of this repository (2026-10-18):** `find . -iname "*seg*"`
returns **no files**, and a repo-wide grep for `segment`/`segmentor` outside the
audit docs matches nothing. So there is no segmentor here to modify, and no
interface of yours to preserve. Everything below is the contract AFR-Net
*offers*; it cannot be validated against your implementation until you supply
it.

`tests/test_afrnet_modules.py::test_no_module_references_a_segmentor` asserts
that nothing under `models/` imports or calls a segmentor, as a standing
regression guard.

---

## 1. How segmentation enters AFR-Net

Through **two optional keyword arguments** of `AFRNet.forward`, and nowhere
else. Both default to `None`, and AFR-Net runs fully without them — tested by
`test_afrnet_runs_without_any_segmentation_input`.

```python
out = model(
    images,                      # [B, 3, 224, 224]
    tokens,                      # [B, L]
    patch_mask=...,              # optional [B, 196] bool
    region_prior_external=...,   # optional [B, 196] float
)
```

Masks are **never fabricated**. With no segmentation supplied, DDTP treats all
196 patches as eligible and ROAM-Med's soft attention prior falls back to DDTP's
own relevance scores.

---

## 2. `patch_mask` — hard eligibility

| Property | Requirement |
|---|---|
| Shape | `[B, 196]` |
| Dtype | `bool` (or anything where `!= 0` means valid) |
| Semantics | `True` = patch **may** be selected; `False` = excluded |
| Coordinate system | row-major over the **14×14** patch grid: index `i` is row `i // 14`, column `i % 14`, matching `models/baseline.py::get_patch_tokens` after the CLS token is dropped |
| Ordering | must match the patch-token order exactly; do **not** re-sort |
| Degenerate case | if fewer than `k` patches are valid, the extra slots are flagged `mask=False` and zeroed rather than filled with padding masquerading as regions |

Use it to confine region selection to anatomy/pathology and exclude background,
borders or burnt-in annotation.

---

## 3. `region_prior_external` — soft emphasis

| Property | Requirement |
|---|---|
| Shape | `[B, 196]` |
| Dtype | float |
| Range | ideally `[0, 1]`. Values outside are passed through a sigmoid, so raw logits are acceptable but less predictable |
| Semantics | larger = more diagnostically relevant |
| Normalisation | if `roam.prior_normalise: true`, rescaled to mean 1 over valid regions, so the prior reweights `S_MR` without changing its expected magnitude |

AFR-Net gathers this at the DDTP-selected indices
(`prior = region_prior_external.gather(1, sel.indices)`) and uses it as
ROAM-Med's soft attention prior in IFA Eq (19), **replacing** DDTP's own scores.

---

## 4. Converting a pixel mask to patch space

If your segmentor emits a pixel mask rather than per-patch values, downsample
to the patch grid. For a 224×224 mask and ViT-B/16:

```python
import torch.nn.functional as F

# mask: [B, 1, 224, 224] float in [0, 1]
coverage = F.avg_pool2d(mask, kernel_size=16)        # [B, 1, 14, 14]
region_prior = coverage.flatten(1)                   # [B, 196], row-major
patch_mask = region_prior > 0.25                     # choose your threshold
```

`avg_pool2d` with `kernel_size=16` gives each patch the **fraction of its area**
covered by the mask, which is the natural soft prior. The flatten is row-major,
matching §2.

Resize before pooling if your mask is not 224×224:
`F.interpolate(mask, size=(224, 224), mode="bilinear", align_corners=False)`.

---

## 5. Keeping the segmentor separate

1. **Do not import it from `models/`.** Call it in your training/eval script,
   convert its output to the tensors above, and pass them in.
2. **Run it under `torch.no_grad()`** unless you intend to train it jointly.
   AFR-Net does not back-propagate into `patch_mask` (boolean) and treats
   `region_prior_external` as data — gradients would flow into a prior that
   requires grad, so detach it if that is not wanted.
3. **Keep its checkpoints separate.** `checkpoints/afrnet/` holds AFR-Net
   module weights only; the frozen backbone and the segmentor are not saved
   there.

Sketch:

```python
# your script — NOT inside models/
seg = load_your_segmentor(...)          # untouched, external
seg.eval()

for images, tokens in loader:
    with torch.no_grad():
        pixel_mask = seg(images)                       # your format
    prior = F.avg_pool2d(pixel_mask, 16).flatten(1)    # [B, 196]
    out = model(images, tokens,
                patch_mask=prior > 0.25,
                region_prior_external=prior)
```

---

## 6. Before relying on this

- **Confirm the grid.** 196 patches row-major on a 14×14 grid is an
  **inference** from `image_size: 224` and `vit_base_patch16_224`, not yet
  observed at runtime. Verify `get_patch_tokens(x).shape == (B, 196, 768)` and
  check that a deliberately one-sided mask steers selection to the expected
  side before trusting the coordinate convention.
- **Check orientation.** A transposed mask is silent and plausible-looking.
  Test with an asymmetric mask (e.g. left half only) and assert the selected
  indices satisfy `i % 14 < 7`.
- **Watch the interaction with WARN-DDTP-1**: a hard `patch_mask` further
  reduces the already non-differentiable selection's flexibility.
