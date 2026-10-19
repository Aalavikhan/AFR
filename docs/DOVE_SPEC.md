# DOVE — Paper-Derived Specification (Source A)

Extracted **verbatim** from the PDF in the repository root:

> Qing Ma, Jiancheng Pan, Cong Bai, *"Direction-Oriented Visual-semantic
> Embedding Model for Remote Sensing Image-text Retrieval"*,
> **IEEE Transactions on Geoscience and Remote Sensing, Vol. 62, 2024.**
> Received 12 Oct 2023; accepted 20 Apr 2024; published 23 Apr 2024.
> File: `Direction-Oriented Visual-semantic Embedding.pdf` (14 pages, 12 MB).

Text was extracted with PyMuPDF and read directly. Every equation below was
transcribed from §III of that file — **not** from memory or secondary sources.
Equation numbers are the paper's own.

**Scope note.** This file records what the paper says. It does **not** describe
this repository's modules. ROAM-Med, GTGA, DDTP and OT-Align are the user's
biomedical designs; see [MODULE_SPEC.md](MODULE_SPEC.md). DDTP and OT-Align
have **no counterpart in DOVE at all**.

---

## 1. Model overview (§III)

Four parts: (1) input representation, (2) **DTGA** — Digging Text Genome
Assistant, (3) **ROAM** — Regional-Oriented Attention Module, (4) objective
function. `d` is the shared embedding size.

---

## 2. Input representation (§III-A)

### 2.1 Visual (§III-A-1)

- **MSV encoder**: multiscale visual features from a ResNet-50 pretrained on
  AID. `M_v = [v_1 … v_{Nm}]ᵀ ∈ R^{Nm×d}`.
- **RoI encoder**: salient-region features, ResNet-50 backbone.
  `R_v = [u_1 … u_{Nr}]ᵀ ∈ R^{Nr×(d/2)}`.
- For a 256×256 image: **`Nm = 4`** ("the semantic-level features from layer1,
  layer2, layer3, layer4 of the ResNet-50") and **`Nr = 36`**.

```
(1)   F_M = MLP(M_v) + M_v                      F_M ∈ R^{Nm×d}
(2)   F_R = R_v W_r + b_r                       F_R ∈ R^{Nr×d}
```

So the multiscale branch is a residual MLP; the region branch is a single
fully-connected layer that also lifts `d/2 → d`.

### 2.2 Textual (§III-A-2)

One-hot → 300-d GloVe embeddings `e_i = W_e w_i` → bidirectional GRU:

```
(3)   h^f_i = GRU_f(e_i, h^f_{i-1}),   H^f = [h^f_1 … h^f_{Nc}]ᵀ
(4)   h^b_i = GRU_b(e_i, h^b_{i+1}),   H^b = [h^b_1 … h^b_{Nc}]ᵀ
(5)   F_G   = DTGA(H^f, H^b)                    F_G ∈ R^{Nc×d}
```

The paper stresses that averaging `H^f` and `H^b` (as prior work does) "can
damage the text semantic content" — DTGA exists to replace that average.

---

## 3. Contextual enhancement (§III-B)

### 3.1 Gated Self-Attention, `G(·)` (§III-B-1, Eq 6–11)

Input `X ∈ R^{Nc×d}`, output `X̃ ∈ R^{Nc×d}`.

```
(6)   X̃ = G(X)

(7)   Q = X W_Q + b_Q
      K = X W_K + b_K
      V = X W_V + b_V                           all ∈ R^{Nc×d}

(8)   G = σ((Q ⊙ K) W_A + b_A)                  G ∈ R^{Nc×d}
(9)   Q′ = G ⊙ Q
(10)  K′ = G ⊙ K
(11)  X̃ = Softmax(Q′K′ᵀ / √d) V
```

**Fidelity-critical details:**

- `⊙` is the **element-wise** product. Eq (8) multiplies Q and K element-wise,
  passes the result through **one fully-connected layer**, then a **sigmoid**.
  This is *not* a standard gate on the input, and *not* a dot product.
- The gate multiplies **Q and K only** — `V` is left ungated (Eq 9–10).
- Eq (11) is otherwise standard scaled dot-product attention; softmax is
  row-wise.
- §IV-B: "In the GA module, there are **2 headers** for text feature encoding."
  So the implementation is 2-head, which Eq (11) alone does not show.

### 3.2 DTGA (§III-B-2, Eq 12–16)

Fig. 3 shows backward features mining forward features.

```
(12)  H̃^f = G(H^f)                              gated self-attention, Eq 6
(13)  T^f  = H̃^f + H^f                          forward joint features
(14)  P^b  = F(G(H^b))                          F(·) = 2-layer fully connected
(15)  T^{f⊙b} = T^f ⊙ P^b                       interactive features
      ... symmetrically, T^{b⊙f}
      T^c = T^{f⊙b} + T^{b⊙f}
(16)  F_G = MLP(T^c) + T^c                      F_G ∈ R^{Nc×d}
```

**Fidelity-critical details:**

- The two directions are **asymmetric in role but symmetric in form**: each
  direction's *joint* feature (Eq 13) is modulated by the *other* direction's
  *probability matrix* (Eq 14), then the two interactions are summed.
- Eq (14) applies gated self-attention **and then** a 2-layer FC. The paper
  calls `P^b` a "backward probability matrix" obtained "by global augmentation
  and nonlinear mapping".
- **UNSPECIFIED in the paper:** the output activation of `F(·)`. The name
  "probability matrix" and the multiplicative use in Eq (15) imply a bounded
  non-negative range, but the paper does not state sigmoid. Any implementation
  must flag this as an inference.
- Eq (16) is a residual MLP, mirroring Eq (1).
- §IV-C-2 ablates the DTGA *inputs* (`H^f,H^f` / `H^b,H^b` / `H^f,H^b` /
  averaged), confirming the two inputs are the GRU direction outputs.

---

## 4. ROAM (§III-C)

Two parts, both with "encoding and decoding parts" (Fig. 4):
**IFA** fuses regional + multiscale visual features; **IGA** uses regional
visual features to guide textual features.

### 4.1 Intra-modal Fusion Attention — IFA (§III-C-1, Eq 17–22)

```
(17)  F′_M = F_M W_M + b_M                      ∈ R^{Nm×d}
(18)  F′_R = F_R W_R + b_R                      ∈ R^{Nr×d}
(19)  S_MR = σ(F′_M F′_Rᵀ)                      ∈ R^{Nm×Nr}
(20)  F_{R⊕M} = S_MR F′_R + F′_M                ∈ R^{Nm×d}
(21)  F_{M⊕R} = S_MRᵀ F′_M + F′_R               ∈ R^{Nr×d}
(22)  F_MR = Concat(F_{R⊕M}, F_{M⊕R}) W_L + b_L ∈ R^{(Nm+Nr)×d}
```

**Fidelity-critical details:**

- **Eq (19) uses σ (sigmoid), NOT softmax.** The score matrix is an
  element-wise gate in `(0,1)^{Nm×Nr}`, with **no normalisation over either
  axis** and **no `1/√d` scaling**. Replacing this with softmax attention
  changes the module's identity. The paper calls `S_MR` simply "the input
  matrix multiplication result".
- Eq (20)/(21) are **cross-residual**: each side is activated by the *other*
  modality's projected features and then has its *own* projected features
  added back. Note the residual in Eq (20) is `F′_M` (the query side), not
  `F′_R`.
- Eq (22) concatenates along the **token** axis (`Nm + Nr` rows), then applies
  a **linear head** on the feature axis. Fig. 4(a) labels this "Linear Head".

### 4.2 Inter-modal Guidance Attention — IGA (§III-C-2, Eq 23–28)

Pre-processing exists because "Different texts have different lengths, which
cannot directly interact with the fixed-length image features".

With `F_R = [r_1 … r_{Nr}]ᵀ ∈ R^{Nr×d}` and `F_G = [g_1 … g_{Nc}]ᵀ ∈ R^{Nc×d}`:

```
(23)  E_R = (1/Nr) Σ_i r_i ,   E_G = (1/Nc) Σ_i g_i            both ∈ R^d
      expand to E′_R, E′_G ∈ R^{Bc×d}         Bc = text batch size
(24)  F′_R = E′_R W_R + b_R                   ∈ R^{Bc×d}
(25)  F′_G = E′_G W_G + b_G                   ∈ R^{Bc×d}
(26)  S_RG = σ(F′_R F′_Gᵀ)                    ∈ R^{Bc×Bc}
(27)  F_{R⊕G} = S_RG F′_G + F′_G              ∈ R^{Bc×d}
(28)  F_RG = MLP(F_{R⊕G}) + F_{R⊕G}           ∈ R^{Bc×d}
```

**Fidelity-critical details:**

- **IGA is batch-coupled.** Eq (23) pools each modality to a single vector per
  sample, so Eq (26) produces a `Bc × Bc` matrix over the **mini-batch**, and
  Eq (27) mixes every sample's text features with every other's. The paper is
  explicit: "`E′_R ∈ R^{Bc×d}` … where `Bc` is the batch size of text input".
  This means a sample's output **depends on which other samples share its
  batch** — see [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md) for why that is a
  serious concern for retrieval evaluation.
- **Eq (26) again uses σ, not softmax**, and again has no `1/√d`.
- Eq (27)'s residual is `F′_G` — the **text** side, consistent with "guide
  textual feature representation".
- Eq (28) is a **non-linear head** (residual MLP), where Eq (22) was linear.
  Fig. 4 labels them "Linear Head" and "Nonlinear Head" respectively.
- `E_R` is an unweighted mean over all `Nr` regions; the paper applies **no
  saliency prior** here.

---

## 5. Objective function (§III-D, Eq 29–31)

```
(29)  V_M = (1/Nm) Σ_i m_i                     from F_M = [m_1 … m_{Nm}]ᵀ
      similarly T_G, V_MR, T_RG from F_G, F_MR, F_RG

(30)  L(V,T) = Σ_T̂ [α − S(V,T) + S(V,T̂)]₊
             + Σ_V̂ [α − S(V,T) + S(V̂,T)]₊

(31)  L_total = L(V_MR, T_RG) + λ_g · L(V_M, T_G)
```

- `[x]₊ ≡ max(x, 0)`; `S(·,·)` is **cosine** similarity; `V̂`, `T̂` are
  negatives **within the mini-batch**.
- Eq (30) is the **bidirectional triplet ranking loss**, summed over negatives
  (not averaged, as written).
- Eq (31): `L(V_M, T_G)` is the **global visual-semantic constraint**, an
  external constraint ensuring "the original visual and textual semantics
  remain unchanged".

---

## 6. Hyperparameters (§IV-B)

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Initial LR | 2e-4, ×0.7 every 20 epochs |
| Mini-batch | 100 |
| Embedding size `d` | **512** |
| Epochs | 50 |
| GA heads | **2** |
| Backbone | pretrained ResNet, **frozen** |
| Margin `α` (Eq 30) | **0.2** |
| `λ_g` (Eq 31) | **10.0** |
| Hardware | NVIDIA RTX A6000 |
| Seed | fixed |

§IV-C-1: `λ_g` was swept; mR is lowest at 0.01 and peaks at **10.0**, declining
after — "excessive global semantic constraints can have an antagonistic
effect".

Datasets: RSICD (10,921 images × 5 sentences) and RSITMD (4,743 × 5).
Metrics: R@K (K = 1,5,10) and mR (their mean).

---

## 7. What DOVE does **not** contain

Recorded so nothing is later misattributed to this paper:

- **No patch selection / top-k module.** There is no DDTP. Regions come from a
  pretrained RoI detector (`Nr = 36`), not from scoring and selecting patches.
- **No optimal transport.** The objective is triplet ranking + a triplet-based
  global constraint (Eq 30–31). There is no Sinkhorn, no transport plan, no
  entropic regularisation, no OT-Align.
- **No depthwise convolution anywhere.** DTGA uses bi-GRU + gated
  self-attention + FC layers.
- **No Vision Transformer.** The visual backbone is ResNet-50; there are no
  patch tokens, no CLS token, and no 196-patch grid.
- **No soft attention prior** over regions. Eq (23) is a plain mean.
- **No biomedical content.** The domain is remote sensing.
