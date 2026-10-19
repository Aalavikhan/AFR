# AFR-Net — Mathematical Fidelity Audit and Traceability Matrix

Post-implementation audit of ROAM-Med, GTGA, DDTP and OT-Align, performed
against [DOVE_SPEC.md](DOVE_SPEC.md) (paper) and [MODULE_SPEC.md](MODULE_SPEC.md)
(intended design) — **not** against class names or code comments.

Date 2026-10-18. Commit at audit time: `7858ee5` plus the uncommitted work
described here.

> ## Verification status, stated once and plainly
>
> **No code in this audit has been executed.** The user is away from the
> research machine and asked that nothing be run; the only thing executed was
> read-only text extraction from the PDF. Therefore:
>
> - every row below marked `static` was verified by **reading the code against
>   the equation**, line by line;
> - no row is marked `executed`;
> - `tests/test_afrnet_modules.py` exists and encodes the expected values, but
>   **has never been run**. Its pass/fail state is unknown.
>
> A module is called **faithful** here only where the *implemented expression
> matches the authoritative equation symbol for symbol*. That is a static
> claim about the mathematics, not a claim that it runs.

Classification used:

| Tag | Meaning |
|---|---|
| **FAITHFUL** | implemented expression matches the authoritative equation |
| **ADAPT** | input/interface adaptation; the module's own algorithm unchanged |
| **MOD** | algorithmic modification; original behaviour altered |
| **CHOICE** | no authoritative source existed; newly specified here |
| **WARN** | correct per spec, but with a consequence the user must decide on |

---

## 1. Traceability matrix

### 1.1 Gated Self-Attention — `models/modules/gated_attention.py`

| ID | Operation | Source | Original | Implemented | Shapes | Class | Verified |
|---|---|---|---|---|---|---|---|
| FAITH-GA-1 | Q,K,V projection | Eq (7) | `Q=XW_Q+b_Q` etc. | `self.w_q(x)`, `w_k`, `w_v` | `[B,N,d]`→`[B,N,d]` | FAITHFUL | static + `test_gated_self_attention_matches_equations_7_to_11` |
| FAITH-GA-2 | gate | Eq (8) | `G=σ((Q⊙K)W_A+b_A)` | `torch.sigmoid(self.w_a(q * k))` | `[B,N,d]` | FAITHFUL | static; element-wise product and **single** FC both present |
| FAITH-GA-3 | gate applied to Q,K only | Eq (9),(10) | `Q'=G⊙Q`, `K'=G⊙K`, V ungated | `q = gate*q; k = gate*k`; `v` untouched | `[B,N,d]` | FAITHFUL | static + `test_gated_self_attention_leaves_v_ungated` |
| FAITH-GA-4 | scaled dot-product | Eq (11) | `Softmax(Q'K'ᵀ/√d)V` | `(qh@khᵀ)/√head_dim` → softmax → `@vh` | `[B,H,N,N]` | FAITHFUL | static |
| ADAPT-GA-1 | multi-head | §IV-B "2 headers" | 2 heads | `num_heads=2` default; `√d` uses **per-head** `d` | — | ADAPT | static. Eq (11) is written single-head; per-head scaling is the standard reading but the paper does not state it. |
| CHOICE-GA-1 | key masking | — | not discussed | `-inf` on padded keys; all-padding rows neutralised to avoid NaN | `[B,N]` | CHOICE | static + `test_gated_self_attention_masks_padding` |

### 1.2 ROAM-Med / IFA — `models/modules/roam_med.py`

| ID | Operation | Source | Original | Implemented | Shapes | Class | Verified |
|---|---|---|---|---|---|---|---|
| FAITH-IFA-1 | projections | Eq (17),(18) | `F'_M=F_MW_M+b_M`, `F'_R=F_RW_R+b_R` | `self.w_m(f_m)`, `self.w_r(f_r)` | `[B,Nm,D]`,`[B,Nr,D]` | FAITHFUL | static |
| FAITH-IFA-2 | score matrix | Eq (19) | `S_MR=σ(F'_M F'_Rᵀ)` | `torch.sigmoid(f_m_p @ f_r_p.transpose(1,2))` | `[B,Nm,Nr]` | FAITHFUL | static + `test_eq19_is_sigmoid_not_softmax`. **Sigmoid, no softmax, no `1/√d`** |
| FAITH-IFA-3 | converged feature M-side | Eq (20) | `F_{R⊕M}=S_MR F'_R + F'_M` | `s_mr @ f_r_p + f_m_p` | `[B,Nm,D]` | FAITHFUL | static + `test_ifa_matches_equations_17_to_22`; residual is the **query** side |
| FAITH-IFA-4 | converged feature R-side | Eq (21) | `F_{M⊕R}=S_MRᵀ F'_M + F'_R` | `s_mr.transpose(1,2) @ f_m_p + f_r_p` | `[B,Nr,D]` | FAITHFUL | static |
| FAITH-IFA-5 | linear head | Eq (22) | `Concat(·,·)W_L+b_L` | `self.w_l(torch.cat([...], dim=1))` | →`[B,Nm+Nr,D]` | FAITHFUL | static; concat on the **token** axis, linear on the feature axis |
| ADAPT-ROAM-1 | `F_M` source | Eq (1) + §III-A-1 | MSV encoder, 4 ResNet stages, `Nm=4` | `SpatialPyramidPool(scales=(1,2,3,7))` over 196 ViT patches, `Nm=63` | `[B,196,D]`→`[B,63,D]` | **ADAPT** | static. A ViT has no spatial pyramid; see §3.1 |
| ADAPT-ROAM-2 | `F_R` source | Eq (2) + §III-A-1 | RoI detector, `Nr=36` | DDTP top-98 patches | `[B,98,D]` | **ADAPT** | static. No detector available; adding one is forbidden; see §3.2 |
| CHOICE-ROAM-1 | soft attention prior | user requirement; **no prior definition existed** | — | `prior_mode="multiply"`: `S_MR ← σ(·) ⊙ prior`, prior rescaled to mean 1 | `[B,Nr]` | CHOICE | static + `test_ifa_prior_multiply_scales_scores` |
| CHOICE-ROAM-2 | region masking in IFA | — | not discussed | `S_MR` columns and `F_{M⊕R}` rows zeroed for invalid regions | — | CHOICE | static + `test_ifa_region_mask_zeroes_invalid_regions` |

### 1.3 ROAM-Med / IGA

| ID | Operation | Source | Original | Implemented | Shapes | Class | Verified |
|---|---|---|---|---|---|---|---|
| FAITH-IGA-1 | pooling | Eq (23) | `E_R=(1/Nr)Σr_i`, `E_G=(1/Nc)Σg_i` | `masked_mean(...)`; identical to the paper when masks are `None` | →`[B,D]` | FAITHFUL | static + `test_masked_mean_matches_plain_mean_without_a_mask` |
| FAITH-IGA-2 | projections | Eq (24),(25) | `F'_R=E'_RW_R+b_R`, `F'_G=E'_GW_G+b_G` | `self.w_r(e_r)`, `self.w_g(e_g)` | `[B,D]` | FAITHFUL | static |
| FAITH-IGA-3 | score matrix | Eq (26) | `S_RG=σ(F'_R F'_Gᵀ)`, `[Bc,Bc]` | `torch.sigmoid(f_r_p @ f_g_p.transpose(0,1))` | `[B,B]` | FAITHFUL | static + `test_iga_matches_equations_23_to_28_and_is_batch_coupled` |
| FAITH-IGA-4 | guided text feature | Eq (27) | `F_{R⊕G}=S_RG F'_G + F'_G` | `s_rg @ f_g_p + f_g_p` | `[B,D]` | FAITHFUL | static; residual on the **text** side |
| FAITH-IGA-5 | non-linear head | Eq (28) | `MLP(F_{R⊕G})+F_{R⊕G}` | `ResidualMLP` | `[B,D]` | FAITHFUL | static; non-linear here vs linear in Eq (22) — distinction preserved |
| CHOICE-IGA-1 | masked pooling | — | plain mean | padding/invalid regions excluded from Eq (23) | — | CHOICE | static + `test_iga_masked_mean_ignores_padding` |
| MOD-ROAM-1 | `iga_mode="diagonal"` | — | Eq (26) is `[Bc,Bc]` | optional per-sample scalar `σ(F'_R[i]·F'_G[i])` | `[B]` | **MOD** | static + `test_iga_diagonal_mode_is_batch_independent`. **Off by default.** |
| WARN-ROAM-1 | batch coupling | Eq (23),(26),(27) | paper's own behaviour | faithful default | — | **WARN** | static + `test_iga_batch_mode_output_depends_on_other_samples`. See §4.1 |

### 1.4 GTGA — `models/modules/gtga.py`

| ID | Operation | Source | Original | Implemented | Shapes | Class | Verified |
|---|---|---|---|---|---|---|---|
| CHOICE-GTGA-1 | branch operator | user design | depthwise conv for local clinical phrases | `Conv1d(C,C,k,groups=C)` → GELU → `Conv1d(C,C,1)` | `[B,L,C]` | CHOICE | static + `test_gtga_branch_is_actually_depthwise` (asserts `groups==C`, weight `[C,1,k]`) |
| MOD-GTGA-1 | branch semantics | DTGA Eq (12), §III-B-2 | `H^f`,`H^b` = GRU **time directions** (Eq 3/4: forward sees `≤ i`, backward `≥ i`) | `branch_mode="symmetric"` (default): kernel widths 3 vs 5, both bidirectional → mines **width**, **not** direction. `branch_mode="directional"`: causal vs anti-causal depthwise conv → mines **direction**, matching Eq (3)/(4) information flow | — | **MOD** (reducible) | static + `test_gtga_directional_branches_have_opposite_receptive_fields`, `test_gtga_directional_branches_differ_at_the_start_too`, `test_gtga_symmetric_branch_sees_both_sides`. See WARN-GTGA-1 |
| WARN-GTGA-1 | default branch mode | — | — | default `"symmetric"`; **`"directional"` recommended** | — | **WARN** | §4.9 |
| GAP-GTGA-1 | global connectivity | DTGA Eq (12) `G(·)` links arbitrary word pairs; abstract: "enhance global word-level semantic connections" | gated self-attention | depthwise k=3/5 — **cannot** link distant tokens in either branch mode | — | **NOT REPRODUCED** | static. §4.10 |
| FAITH-GTGA-2 | joint feature | Eq (13) | `T^f = H̃^f + H^f` | `t_a = b_a + h` | `[B,L,C]` | FAITHFUL (form) | static |
| FAITH-GTGA-3 | probability map | Eq (14) | `P^b = F(G(H^b))`, `F` = 2-layer FC | `self.probe_b(b_b)` = 2-layer FC | `[B,L,C]` | FAITHFUL (form) | static. `G(·)` omitted, consistent with MOD-GTGA-1 |
| INFER-GTGA-1 | `F(·)` output activation | Eq (14) | **unstated in the paper** | sigmoid (configurable) | — | **INFERENCE** | static. "probability matrix" + multiplicative use in Eq (15) imply bounded non-negative; the paper never says sigmoid |
| FAITH-GTGA-4 | cross mining | Eq (15) | `T^{f⊙b}=T^f⊙P^b`; `T^c` = sum both ways | `t_a*p_b + t_b*p_a` | `[B,L,C]` | FAITHFUL | static |
| MOD-GTGA-2 | extra gate | Eq (8) *form* | gate on Q,K inside `G(·)` | `σ(W_A(T_a⊙T_b))⊙T_c` | `[B,L,C]` | **MOD** | static. Paper's gate *form*, new location |
| FAITH-GTGA-5 | decoder | Eq (16) | `MLP(T^c)+T^c` | `ResidualMLP` | `[B,L,C]` | FAITHFUL | static |
| CHOICE-GTGA-2 | padding hygiene | — | not discussed (GRU, no kernel) | zero invalid before **and** after each conv, and on output | `[B,L]` | CHOICE | static + `test_gtga_padding_does_not_leak_into_valid_positions` |
| CHOICE-GTGA-3 | MLP width/activation | Eq (16) unstated | — | 4× expansion, GELU | — | CHOICE | static |

### 1.5 DDTP — `models/modules/ddtp.py`

No DOVE counterpart; every row is `CHOICE`. Authoritative source is the user's
prompt (§8).

| ID | Operation | Requirement | Implemented | Shapes | Verified |
|---|---|---|---|---|---|
| CHOICE-DDTP-1 | relevance | "conditioned on the appropriate text representation" | `cos(W_p x_j, W_t q)/τ`, `τ=0.07` | `[B,Np]` | static + `test_ddtp_picks_the_highest_scoring_patches` |
| CHOICE-DDTP-2 | conditioning vector | — | masked mean of the **GTGA output** | `[B,D]` | static |
| CHOICE-DDTP-3 | selection size | "98 of 196 at 50 %" | `k=round(ratio·Np)`, clamped | `k=98` | static + `test_ddtp_selects_98_of_196` |
| CHOICE-DDTP-4 | configurability | "configurable counts or ratios" | `ratio` or `num_keep` | — | static + `test_ddtp_configurable_counts` |
| CHOICE-DDTP-5 | index preservation | "preserve the original patch indices" | indices sorted **ascending**; features gathered at them | `[B,k]` | static + `test_ddtp_preserves_original_patch_identity` |
| CHOICE-DDTP-6 | special tokens | "exclude class tokens" | CLS dropped upstream in `get_patch_tokens`; only 196 spatial patches eligible | — | static |
| CHOICE-DDTP-7 | masks / invalid patches | "handle masks, invalid patches" | `-inf` scores; under-filled slots marked invalid and zeroed | `[B,k]` | static + `test_ddtp_never_selects_masked_patches_when_enough_remain`, `test_ddtp_flags_slots_when_valid_patches_are_too_few` |
| CHOICE-DDTP-8 | gradient path | "verify gradient propagation; document non-differentiability" | `features × k·softmax(s_sel)` | `[B,k]` | static + `test_ddtp_gradient_reaches_the_scorer_but_not_the_indices` |
| WARN-DDTP-1 | top-k non-differentiability | inherent | **no gradient through *which* patches are chosen** | — | §4.3 |
| WARN-DDTP-2 | tie-breaking | "handle ties" | output *order* deterministic via sort; *membership* of an exact tie may differ across devices | — | §4.4 |

### 1.6 OT-Align — `models/modules/ot_align.py`

No DOVE counterpart; authoritative source is the user's prompt (§9).

| ID | Item | Implemented | Verified |
|---|---|---|---|
| CHOICE-OT-1 | formulation | entropic Kantorovich OT, `min⟨P,C⟩ − εH(P)` s.t. `P1=a, Pᵀ1=b, P≥0` | static |
| CHOICE-OT-2 | inputs | DDTP regions (default) or all 196 patches × valid GTGA tokens | static |
| CHOICE-OT-3 | cost matrix | `C = 1 − cos(x,y) ∈ [0,2]`; `"euclidean"` = `√(2−2cos)` | static + `test_ot_cost_matrix_is_cosine_distance` |
| CHOICE-OT-4 | normalisation | L2 before the cost (required for cosine) | static |
| CHOICE-OT-5 | marginals | uniform over **valid** entries only | static + `test_ot_excludes_padded_tokens_from_the_plan` |
| CHOICE-OT-6 | padding exclusion | large finite cost on invalid pairs; plan zeroed there | static |
| CHOICE-OT-7 | solver | log-domain Sinkhorn–Knopp via `logsumexp` | static |
| CHOICE-OT-8 | stopping | `max\|Δf\| < 1e-6`, cap 100 iters; iteration count returned | static |
| CHOICE-OT-9 | entropic regularisation | `ε = 0.05` | static |
| CHOICE-OT-10 | scalar loss | `⟨P,C⟩`, mean over batch | static |
| CHOICE-OT-11 | numerical stability | **forces float32** inside the solver regardless of autocast | static; see WARN-OT-1 |
| CHOICE-OT-12 | gradient | `detach_plan=True` → envelope gradient `∂⟨P,C⟩/∂θ`, `P` fixed | static + `test_ot_gradient_flows_into_features` |
| CHOICE-OT-13 | constraint check | `max_marginal_error` returned for assertion | static + `test_ot_plan_satisfies_marginal_constraints` (tol `1e-4`) |
| CHOICE-OT-14 | batch / variable length | per-sample masks; independent marginals | static + `test_ot_handles_variable_lengths_and_stays_finite` |
| WARN-OT-1 | reduced precision | Sinkhorn in fp16/bf16 is unreliable; mitigated by forcing fp32 | §4.5 |
| NOTE-OT-1 | `pot` unused | declared in `pyproject.toml`, not used — reasons in the module docstring | static |

### 1.7 Objective — `models/modules/losses.py`

| ID | Operation | Source | Implemented | Class | Verified |
|---|---|---|---|---|---|
| FAITH-LOSS-1 | triplet ranking | Eq (30) | cosine `S`, margin `α`, `[·]₊`, both directions, diagonal excluded | FAITHFUL | static + `test_triplet_loss_hand_computed` (margin 1.5, B=2 → **1.0** exactly), `test_triplet_loss_excludes_the_positive_pair` |
| FAITH-LOSS-2 | Eq (31) combination | Eq (31) | `dove_total_loss` = `L(V_MR,T_RG) + λ_g L(V_M,T_G)` | FAITHFUL | static |
| FAITH-LOSS-3 | Eq (29) means | Eq (29) | `masked_mean` over `F_M`, `F_MR`, `F_G` | FAITHFUL | static |
| MOD-LOSS-1 | primary objective | Eq (30) is DOVE's primary | **default stays `"clip"`** (existing InfoNCE); `"triplet"` opt-in | **MOD (deliberate)** | static + `test_clip_loss_matches_baseline_formula`. Preserves baseline comparability — see §4.6 |
| MOD-LOSS-2 | reduction | Eq (30) is a **sum** | default `"mean"` (÷ B); `"sum"` available | **MOD** | static. A raw sum is batch-size dependent, which would make `λ` values non-transferable |
| CHOICE-LOSS-1 | `λ_ot` | — | `0.1`, **untuned** | CHOICE | — |
| CHOICE-LOSS-2 | `λ_g` | paper: `10.0` | **`0.0` (disabled)** by default | CHOICE | static. The paper's value was tuned for its own architecture/scale/batch — not transplanted |

---

## 2. Frozen-backbone audit

| Check | Implementation | Verified |
|---|---|---|
| all backbone params frozen | `AFRNet.__init__` → `backbone.freeze_all()` (incl. `logit_scale`) | static + `test_afrnet_backbone_is_frozen_and_gets_no_gradients` |
| no backbone gradients | features extracted under `torch.no_grad()` (`freeze_backbone_grad=True`) | static |
| optimizer excludes backbone | `trainable_parameters()` filters `requires_grad`; `trainable_module_names()` asserts no `backbone.*` | static + `test_afrnet_optimizer_sees_only_module_parameters` |
| modules still get gradients | all four are downstream of the frozen features | static + `test_afrnet_all_four_modules_receive_gradients` |
| backbone architecture unchanged | `models/baseline.py` edits are **additive** (`pad_token_id`, `text_pad_mask`, `encode_word_tokens`, `freeze_all`) plus the AFR-009 bug fix to `get_word_tokens`. Loading strategy, architecture and `clip_loss` untouched. | static, `git diff` |
| padding masks correct | pad id read **from the tokenizer**, not assumed 0 | static |

---

## 3. Adaptations in detail

### 3.1 ADAPT-ROAM-1 — multiscale features

**Incompatible operation.** Eq (1) requires `M_v ∈ R^{Nm×d}` from a ResNet MSV
encoder: one token per stage, each at a different spatial resolution.

**Why faithful reproduction is impossible.** BiomedCLIP's visual tower is a
timm ViT-B/16 (`open_clip_config.json` → `timm_model_name:
vit_base_patch16_224`). A ViT has **no spatial downsampling**: every block
operates on the same 14×14 grid at the same width. There is no stage pyramid to
read, and the segmentor may not be modified to supply one.

**Derivation investigated.** Two routes exist from the frozen features:
(a) **spatial** — adaptive average pooling of the 14×14 patch grid at several
scales; (b) **depth** — mean-pooled tokens from several transformer blocks,
which is arguably closer to the paper's wording ("semantic-level features from
layer1..layer4").

**Chosen.** (a), implemented as `SpatialPyramidPool`. It needs only the 196
patch tokens already exposed, adds no dependency on an unverified timm API
(`get_intermediate_layers`), and is deterministic. `Nm = 63` at scales
(1,2,3,7).

**Preserved.** The `[B, Nm, D]` contract, so Eq (17)–(22) are untouched; and
genuine multi-scale spatial context, which is what IFA consumes.

**Lost.** The paper's `Nm = 4` token count, and *semantic-depth* hierarchy —
pyramid levels differ in receptive field but all come from the final block.
Route (b) remains the better match to the paper's wording and is the obvious
first ablation.

### 3.2 ADAPT-ROAM-2 — regional features

**Incompatible operation.** Eq (2) requires `R_v ∈ R^{Nr×(d/2)}` from an RoI
detector, `Nr = 36`.

**Why impossible.** No detector exists in the repository, BiomedCLIP provides
none, and the user forbids introducing one or fabricating masks.

**Chosen.** DDTP's top-98 text-conditioned patches serve as `F_R`. DDTP is
itself a region selector, so this is the natural substitution, and it keeps the
architecture detector-free.

**Preserved.** The `[B, Nr, D]` contract; "regions" remain a sparse, salient
subset of the image rather than the whole grid.

**Lost.** Regions are **axis-aligned single patches**, not detector boxes of
varying size/aspect, and they are **text-conditioned** rather than
class-agnostic. The latter is a real semantic difference: DOVE's regions are
fixed per image, whereas AFR-Med's change with the query text. That also means
`F_R` is not cacheable per image — see §4.2.

---

## 4. Warnings requiring a decision

### 4.1 WARN-ROAM-1 — faithful IGA couples the mini-batch

Eq (23) pools to one vector per sample, so Eq (26) is `[Bc, Bc]` and Eq (27)
mixes every sample's text with every other's. This is unambiguously the
paper's specification, and it is the default.

**Consequence.** A sample's text embedding **depends on which other samples
share its batch.** Therefore: embeddings are not reproducible across batch
compositions; evaluation results vary with batch size and ordering; and at
batch size 1, `S_RG` is `1×1`, a different function altogether.

For retrieval evaluation this is serious: `evaluators/retrieval_eval.py`
encodes the dataset in batches and compares all pairs, so the embedding matrix
would depend on `DataLoader` batching.

**Options.** (a) keep `"batch"` and fix the eval batch size, documenting it as
part of the protocol; (b) use `"diagonal"` (MOD-ROAM-1), losing fidelity but
gaining batch independence. **This is the user's call.** Recommendation:
`"diagonal"` for any reported metric, `"batch"` only for paper-comparison
ablations — and state which was used.

### 4.2 WARN-EVAL-1 — AFRNet embeddings are cross-conditioned

`RetrievalEvaluator` calls `encode_image(images)` and `encode_text(tokens)`
*independently* (`retrieval_eval.py:59-60`). AFRNet cannot satisfy that
contract: DDTP conditions the visual side on the text query, and IGA conditions
the text side on the regions. Neither embedding exists without the other.

Rather than return something silently unconditioned, both methods **raise
`NotImplementedError`** with an explanatory message.

**Consequence.** Retrieval evaluation needs an `N × N` loop over *pairs*, not
`N` encodings — `O(N²)` forward passes. At `N ≈ 9,900` that is ~98 M forward
passes, which is **not feasible**. This interacts with AFR-006 (the evaluator
already risks OOM on this GPU).

**This is an unresolved architectural issue, not an implementation bug.**
Options: (a) a two-stage retrieve-then-rerank protocol (cheap unconditioned
embeddings shortlist `k`, AFRNet reranks); (b) make the visual side
text-independent (drop DDTP's conditioning) and keep conditioning on the text
side only; (c) accept `O(N²)` on a small evaluation subset and report the
subset size. **The user must choose before any retrieval number is produced.**

### 4.3 WARN-DDTP-1 — top-k is not differentiable

No gradient flows through *which* patches are selected. The scorer learns only
through the score weighting of CHOICE-DDTP-8. If selection quality turns out to
be the bottleneck, a perturbed-top-k or Gumbel relaxation would be needed, which
would be an algorithmic change.

### 4.4 WARN-DDTP-2 — tie-breaking is device-dependent

`torch.topk` breaks exact ties by lower index on CPU, but CUDA gives no such
guarantee. Sorting makes the output *order* deterministic; *membership* of an
exact tie can differ CPU vs GPU. With continuous cosine scores exact ties are
improbable, but not impossible under reduced precision.

### 4.5 WARN-OT-1 — Sinkhorn and reduced precision

Entropic OT with `ε = 0.05` needs a dynamic range that fp16 cannot hold:
`exp(-C/ε)` with `C ∈ [0,2]` spans `e^0` to `e^-40`. The log-domain solver plus
the forced-fp32 cast mitigates this. Under `torch.autocast`, OT-Align will
therefore run at a different precision from the rest of the model — deliberate,
and cheap relative to a ViT forward pass.

### 4.6 WARN-OT-2 — OT-Align sends no gradient to the DDTP scorer

Found during the static self-review, not by a test.

With `ot_source="regions"`, OT-Align receives DDTP's **score-weighted**
features `w ⊙ x` (CHOICE-DDTP-8), but its cost matrix L2-normalises them, and
`normalize(w·x) = normalize(x)` for any `w > 0`. The weighting is therefore
**scale-invariant under the cosine cost**, so `∂L_OT/∂w = 0` and the OT term
contributes **nothing** to training the DDTP relevance function.

This is not a bug — the scorer still receives gradient through the ROAM-Med
path, where `F_R` is consumed unnormalised by Eq (17)–(22). But it does mean
the OT loss cannot be expected to improve patch selection, which is a plausible
reason someone would add it.

**Options:** (a) accept it and rely on the ROAM path (current behaviour);
(b) set `ot_source="patches"` so OT acts on all 196 unweighted patches,
decoupling it from selection entirely; (c) apply the weights *after*
normalisation inside the cost, which changes the cost from a cosine distance to
a weighted one — an algorithmic change needing its own justification.
**Not resolved; the user should pick.**

### 4.7 WARN-PRIOR-1 — the prior's type must be declared, not guessed

An earlier draft decided whether to sigmoid the soft attention prior by
inspecting `prior.min() < 0`. That made the transform **data-dependent**: an
identical prior would be squashed or passed through depending on the batch it
arrived in, which is a silent reproducibility hazard. It now requires an
explicit `prior_is_logits` flag, which `AFRNet` sets from the prior's *source*
(DDTP scores → logits; external segmentor prior → probability). Fixed before
any execution; recorded because the failure mode is instructive.

### 4.9 WARN-GTGA-1 — `branch_mode` decides whether GTGA mines what DTGA mines

Added 2026-10-18 after the question *"will my GTGA function logically exactly
as DTGA does?"*. The honest answer was **no**, and this is the reducible part.

DTGA's two inputs differ by **direction of inference** (Eq 3/4). The original
GTGA branches differed only by **kernel width**, and both were symmetric — so
there was no directional asymmetry for Eq (15) to mine. Worse, k=3 and k=5 over
the same input have heavily overlapping receptive fields, so `B_1` and `B_2`
may be strongly correlated; then `P_a ≈ P_b`, `T_a⊙P_b ≈ T_b⊙P_a`, and the
cross-mining degenerates toward `2 × (one branch modulated by itself)`.

`branch_mode="directional"` fixes this with causal / anti-causal padding on the
depthwise kernel: branch A sees tokens `≤ i`, branch B sees `≥ i`, exactly the
information flow of the forward and backward GRU states. The depthwise design
is fully retained — only the padding changes.

**Default remains `"symmetric"`** so that reviewed behaviour is not changed
silently. **Recommendation: set `branch_mode: directional` with equal kernels
(e.g. `[5, 5]`)** so direction is the only difference between branches, as in
DTGA. The degeneracy above is a *risk*, not a measured fact — it has not been
executed.

### 4.10 GAP-GTGA-1 — DTGA's global connectivity is not reproduced

Neither branch mode recovers it. DTGA's `G(·)` is gated self-attention over the
whole sequence; the abstract's stated purpose is to *"expand the range of
tractable textual representation and enhance global word-level semantic
connections"*, and §III-B-2 claims it captures *"long-term dependencies …
reducing the temporal forgetting problem of GRU structures"*. A k=3/5 depthwise
kernel reaches 1–2 tokens: in a 30-token clinical sentence, tokens 2 and 28
have **no path between them** inside GTGA.

**Why this is still defensible.** The input is PubMedBERT's contextual output —
12 layers of full self-attention — so global context is already present before
GTGA runs, and the GRU forgetting DTGA targets does not exist in a transformer.
GTGA's role is local phrase refinement on top of global features. That is a
coherent design, but it is **a different justification from DTGA's, not the
same mechanism**, and must not be described as reproducing DTGA.

**If longer range is wanted:** dilated depthwise kernels (k=3, dilation 1/2/4)
keep the convolutional design; `use_gated_self_attention=True` restores the
attention path itself but moves GTGA toward DTGA, which the design explicitly
did not want.

### 4.11 MOD-LOSS-1 — primary objective kept as InfoNCE

DOVE's primary objective is Eq (30) triplet ranking. AFR-Net's default stays
the **existing** BiomedCLIP symmetric InfoNCE, because the instruction was to
preserve the current training objective unless there is a documented reason to
change it, and because the recorded baseline (once measured) must stay
comparable. `primary="triplet"` is available and must be reported if used.

---

## 5. What is NOT faithful, summarised

Honest bottom line per module:

| Module | Verdict |
|---|---|
| **Gated Self-Attention** | **Faithful** to Eq (6)–(11). Only multi-head scaling (ADAPT-GA-1) and key masking (CHOICE-GA-1) go beyond the text. |
| **ROAM-Med / IFA** | **Faithful algorithm, adapted inputs.** Eq (17)–(22) transcribed exactly; both inputs are substitutes (ADAPT-ROAM-1/2). Not "faithful ROAM" overall, because IFA's *inputs* are a different representation. |
| **ROAM-Med / IGA** | **Faithful** to Eq (23)–(28) in the default `"batch"` mode, including the batch coupling. |
| **GTGA** | **Custom, not faithful to DTGA, and not claimed to be.** It keeps DTGA's Eq (13)–(16) skeleton but replaces the branch operator with depthwise convolutions and adds a gate at a new location (MOD-GTGA-2). With `branch_mode="directional"` it does mine the same *quantity* DTGA mines (direction of inference, MOD-GTGA-1); with the default `"symmetric"` it mines receptive-field width instead. **Neither mode reproduces DTGA's global connectivity** (GAP-GTGA-1). |
| **DDTP** | **Custom.** No fidelity claim is possible — DOVE has no such module. Faithful to the *user's* stated requirements. |
| **OT-Align** | **Custom.** No fidelity claim is possible. Faithful to the standard entropic-OT formulation it declares. |
| **Objective** | **Faithful components, deliberately different composition** (MOD-LOSS-1/2). |

**Nothing in this repository is entitled to the phrase "faithful
reproduction of DOVE."** The visual representation is fundamentally different
(ViT patches vs ResNet pyramid + RoI boxes), two of the four modules have no
paper counterpart, and the primary loss differs by choice.

---

## 6. Files changed

| File | Change |
|---|---|
| `models/modules/__init__.py` | **new** — package exports + provenance table |
| `models/modules/_common.py` | **new** — masking, pooling, `ResidualMLP`, `TwoLayerFC`, `SpatialPyramidPool` |
| `models/modules/gated_attention.py` | **new** — DOVE Eq (6)–(11) |
| `models/modules/roam_med.py` | **new** — IFA Eq (17)–(22), IGA Eq (23)–(28), prior |
| `models/modules/gtga.py` | **new** — depthwise branches + DTGA Eq (13)–(16) skeleton |
| `models/modules/ddtp.py` | **new** — text-conditioned top-k selection |
| `models/modules/ot_align.py` | **new** — log-domain Sinkhorn + OT loss |
| `models/modules/losses.py` | **new** — Eq (30), Eq (31), composed objective |
| `models/afrnet.py` | **new** — composition of backbone + four modules |
| `models/baseline.py` | **modified** — AFR-009 fix to `get_word_tokens`; added `pad_token_id`, `text_pad_mask`, `encode_word_tokens`, `freeze_all`. Nothing removed or renamed. |
| `configs/afrnet.yaml` | **new** — `configs/baseline.yaml` deliberately untouched |
| `tests/test_afrnet_modules.py` | **new** — ~60 assertions, **not executed** |
| `docs/DOVE_SPEC.md`, `MODULE_SPEC.md`, `FIDELITY_AUDIT.md`, `SEGMENTOR_INTEGRATION.md` | **new** |

**Segmentor regression check.** `find . -iname "*seg*"` returns **no files** —
there is no segmentor in this repository, so none could be modified.
`test_no_module_references_a_segmentor` additionally asserts that nothing under
`models/` imports or calls one. Nothing in `trainers/`, `evaluators/`,
`scripts/`, `tools/` or `configs/baseline.yaml` was changed.

---

## 7. Open items

1. **Run the tests.** `tests/test_afrnet_modules.py` has never executed.
2. **Decide WARN-EVAL-1** — the cross-conditioning / `O(N²)` retrieval problem.
   This blocks producing any retrieval metric.
3. **Decide WARN-ROAM-1** — `iga_mode` for reported results.
4. **Verify the backbone contract at runtime**: that `get_patch_tokens` really
   returns `[B,196,768]` and the fixed `get_word_tokens` really returns
   `[B,L,768]`. Both are inferences from the checkpoint and config.
5. **`data/roco_dataset.py` is still missing** (AFR-001), so nothing runs end
   to end, and `trainers/baseline_trainer.py` has **not** been extended to
   train `AFRNet` — it still trains the baseline only.
6. **Ablate ADAPT-ROAM-1** against the depth-based multiscale variant.
7. **Tune `λ_ot`**; sweep `λ_g` before enabling it.
