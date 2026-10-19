# AFR-Net — Intended Biomedical Module Specification (Source B + C)

What each of the four modules is **meant** to do in this project, as distinct
from what DOVE does ([DOVE_SPEC.md](DOVE_SPEC.md)) and from how faithfully it
was achieved ([FIDELITY_AUDIT.md](FIDELITY_AUDIT.md)).

Written 2026-10-18. **Nothing in this document has been executed.**

---

## 0. Source C was empty

Before implementation, the repository was searched for prior definitions of the
four modules:

```
grep -rin "roam|gtga|ddtp|ot_align|optimal transport|sinkhorn|depthwise|segment" \
     --include=*.py --include=*.yaml --include=*.md --include=*.toml .
```

The only hit outside the audit docs was `train.py:119`,
`print("\nBaseline locked. Ready for AFR-Net modules (Segment 3).")`.

**Consequences, stated plainly:**

- There was **no existing implementation, equation, config block, comment or
  test** for ROAM-Med, GTGA, DDTP or OT-Align. The instruction to "preserve the
  existing design" and "fix bugs without redesigning" had nothing to act on.
- Every design decision below that is not traceable to the DOVE paper or to the
  user's prompt is therefore a **new choice by this implementation**, and is
  labelled `[CHOICE]`. Those are the decisions most worth reviewing.
- **No segmentor exists in this repository.** `find . -iname "*seg*"` returns
  nothing. There was no segmentor file to preserve, modify or avoid. The
  external interface is specified in
  [SEGMENTOR_INTEGRATION.md](SEGMENTOR_INTEGRATION.md).

---

## 1. Backbone and feature interfaces

| Requirement | Status |
|---|---|
| Frozen BiomedCLIP backbone | implemented — `AFRNet.__init__` calls `backbone.freeze_all()`, and features are extracted under `torch.no_grad()` |
| ViT-B/16 image encoder | **[V]** confirmed: `open_clip_config.json` → `timm_model_name: vit_base_patch16_224`, `image_size: 224` |
| PubMedBERT text encoder | **[V]** confirmed: `hf_model_name: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, and the checkpoint's `text.transformer.encoder.layer.{0..11}` keys are a 12-layer BERT |
| 196 spatial patches | **[I]** 224/16 = 14, 14² = 196; `get_patch_tokens` drops the CLS token at index 0. **Not yet observed at runtime.** |
| up to 256 text tokens | **[V]** `context_length: 256` in the config, and open_clip 3.3.0 applies it automatically |
| 768-dim tokens | **[I]** ViT-B/16 and BERT-base both have width 768. The *projected* CLIP space is 512 (`embed_dim: 512`), so **768 is the pre-projection token width** — these are different spaces and must not be conflated. |

### Interface changes made

`models/baseline.py` gained:

- **`get_word_tokens` — rewritten.** The previous body called
  `self.model.text.token_embedding` and `.positional_embedding` and treated the
  text tower as a sequence-first open_clip `TextTransformer`. BiomedCLIP's tower
  is an `HFTextEncoder` wrapping a batch-first HuggingFace BERT, so the old code
  raised `AttributeError` on first use (`KNOWN_ISSUES.md` AFR-009). It now
  passes `input_ids` + `attention_mask` and reads `last_hidden_state`.
- **`pad_token_id()` / `text_pad_mask()`** — pad id resolved *from the
  tokenizer*, not assumed to be 0.
- **`encode_word_tokens()`** — returns `(features, mask)`, the contextual text
  interface the modules consume.
- **`freeze_all()`** — freezes `logit_scale` too. The existing
  `freeze_backbone()` deliberately leaves it trainable (right for CLIP
  fine-tuning, wrong for a frozen-backbone adapter architecture), and was
  **left unchanged**.

Nothing else in `baseline.py` was touched; the architecture, the loading
strategy and `clip_loss` are as they were.

---

## 2. ROAM-Med

**Intent.** DOVE's ROAM, operating on biomedical features, plus a soft
attention prior that emphasises relevant image regions.

**Mathematics: preserved from the paper.** IFA (Eq 17–22) and IGA (Eq 23–28)
are transcribed operation by operation in `models/modules/roam_med.py`,
including the two details most at risk of being "modernised" away:

- **Eq (19) / (26) use sigmoid, not softmax**, with no `1/√d` scaling. These are
  element-wise gates, not normalised attention distributions.
- **Eq (20) / (21) residuals are cross-wired**, and **Eq (22) is a linear head
  while Eq (28) is a non-linear one.**

**Inputs: adapted, because the originals do not exist in BiomedCLIP.**

| DOVE | AFR-Med | Why |
|---|---|---|
| `F_M`, MSV encoder over 4 ResNet stages, `Nm = 4` | `SpatialPyramidPool` over the 196 patch tokens, `Nm = 63` at scales (1,2,3,7) | A ViT has no spatial pyramid. Multi-scale average pooling of the 14×14 grid is the least invasive way to produce a genuine multi-scale token set from the frozen features. `[CHOICE]` of scales. |
| `F_R`, RoI detector, `Nr = 36` | DDTP-selected top-98 patches | No detector is available and adding one is forbidden. DDTP *is* a region selector, so its output is the natural `F_R`. |

Both substitutes honour the `[B, Nm, D]` / `[B, Nr, D]` contract, so Eq (17)–(28)
run unmodified.

**Soft attention prior `[CHOICE]`** — the user required it; no prior definition
existed, so its mathematics is defined here:

- **Entry point:** IFA Eq (19), the only place a per-region weighting can act on
  both Eq (20) and Eq (21) at once.
- **Default (`prior_mode="multiply"`):** `S_MR ← σ(F'_M F'_Rᵀ) ⊙ prior`,
  broadcast over the `Nm` axis. Chosen because Eq (19) is already a gate in
  `(0,1)`; multiplying by a second non-negative gate preserves that semantics,
  whereas adding would not.
- **Alternative (`"add_logit"`):** `S_MR ← σ(logits + log prior)` — the prior
  biases the pre-sigmoid score.
- **Normalisation:** rescaled to mean 1 over valid regions, so applying the
  prior does not change the expected magnitude of `S_MR`.
- **Source of the prior:** DDTP's own relevance scores by default; or an
  external `region_prior_external` (the segmentor hook). Never fabricated.

---

## 3. GTGA

**Intent.** A **custom** depthwise-convolutional module modelling local
clinical phrases over frozen PubMedBERT contextual tokens. Explicitly *not* a
claim about DTGA.

**Design.** GTGA keeps DTGA's dual-branch *mining skeleton* — Eq (13)–(16) — and
replaces the branch *source*:

| DTGA | GTGA |
|---|---|
| `H^f`, `H^b` from a bi-GRU (two **time directions**) | `B_1`, `B_2` from two depthwise-separable conv branches — see `branch_mode` below |
| Eq (12) `H̃^f = G(H^f)` — gated self-attention | `B_k = pointwise(act(depthwise_k(H)))` `[CHOICE]` |
| Eq (13) `T^f = H̃^f + H^f` | `T_a = B_1 + H` — same form |
| Eq (14) `P^b = F(G(H^b))` | `P_2 = F(B_2)`, `F` = 2-layer FC — same form |
| Eq (15) `T^{f⊙b} = T^f ⊙ P^b`, summed both ways | identical |
| Eq (16) `F_G = MLP(T^c) + T^c` | identical |

Keeping the skeleton means the "mine one representation using the other"
interaction survives. What the two branches *differ by* is set by
`branch_mode`, and this is the decision that determines whether GTGA mines the
same thing DTGA mines:

| `branch_mode` | Branch A / B | What Eq (15) mines | Matches DTGA? |
|---|---|---|---|
| `"symmetric"` (default) | both symmetric, k=3 / k=5 | receptive-field **width** | **No** |
| `"directional"` (**recommended**) | **causal** (sees `≤ i`) / **anti-causal** (sees `≥ i`) | direction of **inference** | **Yes**, in the one respect that matters |

`"directional"` reproduces the information flow of DOVE Eq (3)/(4) — the
forward GRU state at `i` depends on tokens `≤ i`, the backward state on tokens
`≥ i` — using causal and anti-causal padding on the depthwise kernel. That
restores the asymmetry §III-B-2 says DTGA exists to exploit: *"Different words
in a sentence have different probabilities of inference in the forward and
backward directions… it is necessary to mine the key features from the other
direction features."*

**Why `"symmetric"` is weaker, not just different.** Both branches see left and
right, and k=3 and k=5 have heavily overlapping receptive fields over the same
input, so `B_1` and `B_2` may be strongly correlated. Then `P_a ≈ P_b`, so
`T_a⊙P_b ≈ T_b⊙P_a`, and Eq (15) degenerates toward `2 × (one branch modulated
by itself)` — paying for two branches and getting roughly one. With
`"directional"`, the two views are complementary by construction.

**Default is `"symmetric"`** only so that behaviour already reviewed is not
changed silently. With `"directional"`, prefer **equal** kernel sizes (e.g.
`[5, 5]`) so direction is the sole difference, as it is in DTGA.

**What neither mode recovers:** DTGA's **global** word-level connectivity.
`G(·)` is gated self-attention linking arbitrary word pairs; a k=3/5 depthwise
kernel cannot connect distant tokens. This is defensible here — PubMedBERT is
12 layers of full self-attention, so the input is *already* globally
contextual, and DTGA's stated target (GRU temporal forgetting) does not apply
to a transformer. But it is a **different justification** from DTGA's, not the
same mechanism. For longer range without attention, use dilated kernels; to
recover the attention path itself, `use_gated_self_attention=True`.

**Depthwise specifics `[CHOICE]`** — `Conv1d(C, C, k, groups=C)` (weight shape
`[C,1,k]`), then GELU, then a pointwise `Conv1d(C, C, 1)`: a depthwise-separable
convolution. `groups=C` is what makes it depthwise; an ordinary `Conv1d` here
would be a different module, and a test asserts `groups == C`.
`padding = k//2` with odd `k` preserves sequence length.

**Gating `[CHOICE]`** — `use_eq8_gate=True` applies DOVE Eq (8)'s *functional
form*, `σ((T_a ⊙ T_b) W_A + b_A)`, to the joint features. The form is borrowed;
the location is new (MOD-GTGA-2).

**Padding.** Invalid positions are zeroed before *and* after every convolution,
and on the module output. A width-5 kernel reaches 2 neighbours, so without
this, padded slots would contaminate real clinical tokens — tested directly.

**`use_gated_self_attention`** exists but defaults to **False**, honouring the
instruction that GTGA must not be replaced by gated self-attention.

---

## 4. DDTP

**Intent.** Custom text-conditioned patch selection: top **98** of **196**
spatial patches at a 50 % retention ratio. No DOVE counterpart.

**All of the following is `[CHOICE]`** — no prior specification existed.

1. **Relevance function.** Project patches and the text query into a shared
   space, L2-normalise both, take the cosine, divide by `tau = 0.07`:
   `s_j = cos(W_p x_j, W_t q) / τ`. Cosine because it is scale-invariant and
   matches the contrastive geometry the backbone was trained in.
2. **Conditioning representation.** The masked mean of the **GTGA output**, so
   selection is conditioned on phrase-aware text rather than raw BERT tokens.
3. **Selection.** `k = round(ratio × Np)`, clamped to `[1, Np]`; `num_keep`
   overrides. Invalid patches get `-inf` and can never be selected.
4. **Index preservation.** Selected indices are **sorted ascending**, so spatial
   order survives and `features[b, i]` corresponds to `patches[b, indices[b, i]]`
   exactly — asserted by a test.
5. **Special tokens.** CLS is already dropped by `get_patch_tokens`, so only the
   196 spatial patches are eligible.
6. **Too-few-valid-patches.** Slots backed by an invalid patch are marked
   `mask=False` and zeroed, rather than passed downstream as if they were
   regions.
7. **Gradient.** `torch.topk` returns discrete indices, so **no gradient flows
   through which patches were chosen.** Gradient reaches the scorer only via
   score weighting: `features × k·softmax(s_sel)` (sums to `k`, so the feature
   scale is preserved in expectation). `score_weighting="none"` starves the
   scorer entirely and exists only for ablation — a test documents that.
8. **Ties.** `topk`'s tie-break is not guaranteed identical across devices.
   Sorting makes the *output order* deterministic; *which* member of an exact
   tie is kept may still vary by device. Documented, not hidden.

---

## 5. OT-Align

**Intent.** Custom optimal-transport alignment between visual regions and valid
text tokens. No DOVE counterpart — the paper has no transport problem at all.

**Formulation `[CHOICE]`** — entropic-regularised Kantorovich OT:

```
C_ij = 1 - cos(x_i, y_j)              in [0, 2]
a    = uniform over valid patches     (sums to 1)
b    = uniform over valid tokens      (sums to 1)
min_P <P, C> - eps*H(P)   s.t.  P1 = a,  Pᵀ1 = b,  P >= 0
L_OT = <P, C>
```

- **Solver:** log-domain Sinkhorn–Knopp, pure PyTorch, `eps = 0.05`,
  `max_iters = 100`, stopping when `max|Δf| < 1e-6`.
- **Why log-domain:** the naive kernel `exp(-C/eps)` underflows to 0 at small
  `eps` in reduced precision, after which the scaling vectors go NaN.
  `logsumexp` is stable.
- **Why not POT:** `pot` is a declared dependency but is **unused** — the loss
  must be differentiable, batched, device-resident and masked per sample.
- **Precision:** the solver forces **float32** internally regardless of
  autocast, then casts the scalar back. Sinkhorn under fp16/bf16 is unreliable.
- **Masking:** any pairing touching an invalid patch or padded token is given a
  large finite cost and is zeroed out of the plan; marginals count valid
  entries only.
- **Gradient:** `detach_plan=True` uses the envelope-theorem gradient
  `∂⟨P,C⟩/∂θ` with `P` held fixed — cheaper and more stable than
  back-propagating through the iteration, and exact at the unregularised
  optimum.
- **Constraint verification:** `max_marginal_error` is returned so a test can
  assert `‖P1 − a‖∞` and `‖Pᵀ1 − b‖∞` are within tolerance (`1e-4` used).

---

## 6. Objective

```
L_total = L_primary(V_MR, T_RG) + lambda_ot * L_OT + lambda_g * L(V_M, T_G)
```

- **`L_primary` defaults to `"clip"`** — BiomedCLIP's symmetric InfoNCE,
  i.e. the objective `trainers/baseline_trainer.py` already optimises. The
  existing primary objective is therefore **preserved, not replaced**, so
  baseline numbers remain comparable. `primary="triplet"` switches to DOVE
  Eq (30) and is an explicit, logged decision.
- **`L_OT`** is auxiliary, `lambda_ot = 0.1` `[CHOICE, UNTUNED]`.
- **`L(V_M, T_G)`** is DOVE's Eq (31) global visual-semantic constraint, with
  its **own** coefficient, defaulting to `lambda_g = 0.0` (disabled). The
  paper's `10.0` was tuned for its architecture, loss scale and batch size 100;
  transplanting it unexamined would be unjustified.
- Eq (29) means are computed with masked pooling so padding cannot dilute them.

---

## 7. Known limitations of this specification

1. **Nothing has been executed.** No test has run; no shape, gradient or
   numerical claim here is confirmed at runtime.
2. **196 patches and 768 dims are inferred**, not observed, because loading the
   backbone needs `transformers` (absent) and ~784 MB from the Hub.
3. **`Nm = 63 ≠ 4`.** The pyramid substitute does not reproduce the paper's
   token count, only its multi-scale character.
4. **Faithful IGA is batch-coupled**, which has consequences for evaluation —
   see WARN-ROAM-1 and WARN-EVAL-1 in [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md).
5. **The pipeline still cannot run end to end**: `data/roco_dataset.py` is
   still missing (AFR-001), and `trainers/baseline_trainer.py` has not been
   extended to train AFRNet.
