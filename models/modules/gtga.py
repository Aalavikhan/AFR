"""
models/modules/gtga.py
-----------------------
GTGA — Gated Text Genome Assistant (this project's **custom** text module).

GTGA is NOT a DOVE module. DOVE's text module is DTGA (§III-B-2, Eq 12-16),
which uses a bidirectional GRU plus gated self-attention. The user's GTGA
instead uses **depthwise convolutions to model local clinical phrases**, over
the frozen PubMedBERT contextual token features.

What is borrowed and what is custom
-----------------------------------
GTGA keeps DTGA's **dual-branch mining skeleton**, because that skeleton is
exactly the "mine one representation using the other" interaction the user
asked to retain, and substituting the branch *source* is an input adaptation
rather than an algorithm change:

    DOVE DTGA                           GTGA (this file)
    ------------------------------      --------------------------------------
    H^f, H^b from a bi-GRU              B_1, B_2 from two depthwise conv
                                        branches with different kernel widths
    (12) H~^f = G(H^f)   (gated attn)   B_k = DWSepConv_k(H)        [CUSTOM]
    (13) T^f = H~^f + H^f               T_a = B_1 + H               [faithful form]
    (14) P^b = F(G(H^b))                P_2 = F(B_2)                [faithful form]
    (15) T^{f(*)b} = T^f (*) P^b        T_12 = T_a (*) P_2          [faithful form]
         T^c = T^{f(*)b} + T^{b(*)f}    T_c  = T_12 + T_21          [faithful]
    (16) F_G = MLP(T^c) + T^c           F_G  = MLP(T_c) + T_c       [faithful]

What distinguishes the two branches depends on `branch_mode`:

* `"symmetric"` (default) - both branches are symmetric convolutions differing
  only in kernel **width** (3 vs 5). This mines receptive-field scale. It is
  **not** what DTGA mines, and because the two branches overlap heavily they
  may carry correlated information, in which case Eq (15)'s cross-mining
  degenerates toward `2 x (one branch modulated by itself)`.
* `"directional"` (**recommended**) - branch A is **causal** (position `i` sees
  tokens `<= i`) and branch B **anti-causal** (sees tokens `>= i`), matching the
  information flow of DOVE Eq (3)/(4)'s forward and backward GRUs. This
  restores the asymmetry DTGA actually exploits - *"Different words in a
  sentence have different probabilities of inference in the forward and
  backward directions"* (§III-B-2) - while keeping the depthwise design.

Either way the branch *operator* is a convolution rather than a GRU plus gated
self-attention, so GTGA still does not provide DTGA's **global** word-level
connectivity: a k=3/5 kernel cannot link distant tokens. That is acceptable
here only because the input is already contextual (PubMedBERT is 12 layers of
full self-attention), which is a different justification from DTGA's. Recorded
as MOD-GTGA-1 in docs/FIDELITY_AUDIT.md.

Two things the paper leaves unspecified, flagged as inferences:
  * the output activation of `F(·)` in Eq (14) — see `TwoLayerFC`;
  * the hidden width / activation of the Eq (16) MLP — see `ResidualMLP`.

Padding correctness
-------------------
A depthwise kernel of width `k` mixes up to `k-1` neighbours, so a non-zero
padded position would contaminate real clinical tokens. Every branch therefore
zeroes invalid positions **before** and **after** each convolution, and the
module zeroes its own output. Tested by
`tests/test_afrnet_modules.py::test_gtga_padding_does_not_leak_into_valid_positions`
and `::test_gtga_directional_padding_does_not_leak`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import ResidualMLP, TwoLayerFC, _activation, zero_invalid
from .gated_attention import GatedSelfAttention


class DepthwiseSeparableBranch(nn.Module):
    """One GTGA branch: depthwise Conv1d -> activation -> pointwise Conv1d.

    * **Depthwise**: `Conv1d(C, C, k, groups=C)` — one kernel per channel, so
      each feature dimension is filtered independently along the token axis.
      `groups=C` is what makes it depthwise; a plain `Conv1d` here would be an
      ordinary convolution and would change the module's identity.
    * **Pointwise**: `Conv1d(C, C, 1)` mixes channels at a single position.
      Together these form a depthwise-separable convolution.

    `direction` controls which tokens position `i` may see. Padding is applied
    manually (the conv itself uses `padding=0`) because `nn.Conv1d` can only
    pad symmetrically:

        "symmetric" - pad (k//2, k//2); position i sees [i-k//2, i+k//2]
        "forward"   - pad (k-1, 0);     position i sees [i-k+1, i]   (causal)
        "backward"  - pad (0, k-1);     position i sees [i, i+k-1]   (anti-causal)

    All three keep the sequence length at `L` for odd `k`.

    The directional modes are the depthwise analogue of DTGA's bi-GRU:
    DOVE Eq (3) defines `h^f_i = GRU_f(e_i, h^f_{i-1})`, i.e. the forward state
    at `i` depends on tokens `<= i`, and Eq (4)'s backward state depends on
    tokens `>= i`. A causal / anti-causal convolution pair reproduces exactly
    that information-flow asymmetry while keeping the convolutional design.

    Input/output are [B, L, C]; the Conv1d layout [B, C, L] is handled inside.
    """

    _DIRECTIONS = ("symmetric", "forward", "backward")

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        activation: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0,
        direction: str = "symmetric",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd so that padding preserves length, got {kernel_size}"
            )
        if direction not in self._DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self._DIRECTIONS}, got {direction!r}"
            )
        self.kernel_size = kernel_size
        self.direction = direction

        k = kernel_size
        if direction == "symmetric":
            self._pad = (k // 2, k // 2)
        elif direction == "forward":
            self._pad = (k - 1, 0)
        else:                                  # "backward"
            self._pad = (0, k - 1)

        # padding=0: the padding above is applied explicitly in forward().
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=0, groups=dim, bias=bias,
        )
        self.act = _activation(activation)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Zero pads BEFORE the convolution: otherwise padded values are mixed
        # into neighbouring valid positions by the depthwise kernel.
        x = zero_invalid(x, mask)
        h = x.transpose(1, 2)                 # [B, C, L]
        h = F.pad(h, self._pad)               # zero padding, direction-aware
        h = self.depthwise(h)
        h = self.act(h)
        h = self.pointwise(h)
        h = h.transpose(1, 2)                 # [B, L, C]
        h = self.drop(h)
        # Zero pads AFTER as well, so the branch output is clean for Eq (13)-(15).
        return zero_invalid(h, mask)


class GTGA(nn.Module):
    """Gated Text Genome Assistant.

    Args:
        dim:            feature size (768 for PubMedBERT token features).
        kernel_sizes:   one odd kernel per branch. Exactly 2 branches are
                        required to mirror DTGA's dual-branch mining.
                        Default (3, 5) = short and longer clinical phrases.
        branch_mode:    what distinguishes the two branches.
                        "symmetric" (default) - both branches are symmetric
                            convolutions differing only in kernel WIDTH. This
                            mines receptive-field scale, not direction.
                        "directional" - branch A is causal (sees tokens <= i),
                            branch B anti-causal (sees tokens >= i), both with
                            their own kernel size. This restores DTGA's actual
                            logic: the two views differ by DIRECTION OF
                            INFERENCE, which is the asymmetry Eq (15) mines.
                            Recommended; see MOD-GTGA-1 in FIDELITY_AUDIT.md.
                            With this mode, consider EQUAL kernel sizes (e.g.
                            (5, 5)) so direction is the only difference between
                            the branches, as it is in DTGA.
        activation:     branch activation.
        probe_activation: output activation of `F(.)` (DOVE Eq 14). "sigmoid"
                        makes `P` a bounded multiplicative probability, which
                        the paper implies but does not state.
        mlp_hidden_mult / mlp_activation: Eq (16) residual MLP.
        use_eq8_gate:   additionally apply a gate of the DOVE Eq (8) functional
                        form, `sigma((T_a (*) T_b) W_A + b_A)`, to `T_c`. This
                        reuses the paper's gate *form* at a different location,
                        and is recorded as MOD-GTGA-2. Default True because the
                        user's design explicitly asks for gating.
        use_gated_self_attention: prepend DOVE's `G(.)` (Eq 6-11) to each
                        branch input, recovering DTGA's global-context path.
                        Default **False** — the user instructed that GTGA must
                        not be replaced by gated self-attention. Opt-in only.

    Shapes:
        tokens: [B, L, dim]
        mask:   [B, L] bool, True = valid token (optional)
        return: [B, L, dim], padded positions exactly zero
    """

    def __init__(
        self,
        dim: int = 768,
        kernel_sizes: Sequence[int] = (3, 5),
        branch_mode: str = "symmetric",
        activation: str = "gelu",
        probe_activation: str = "sigmoid",
        probe_hidden_mult: float = 1.0,
        mlp_hidden_mult: float = 4.0,
        mlp_activation: str = "gelu",
        dropout: float = 0.0,
        use_eq8_gate: bool = True,
        use_gated_self_attention: bool = False,
        gsa_heads: int = 2,
    ):
        super().__init__()
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if len(kernel_sizes) != 2:
            raise ValueError(
                "GTGA mirrors DTGA's dual-branch mining and needs exactly 2 kernel "
                f"sizes, got {kernel_sizes!r}"
            )
        if branch_mode not in ("symmetric", "directional"):
            raise ValueError(
                f"branch_mode must be symmetric/directional, got {branch_mode!r}"
            )
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        self.branch_mode = branch_mode
        self.use_eq8_gate = use_eq8_gate
        self.use_gated_self_attention = use_gated_self_attention

        # In "directional" mode branch A is causal and branch B anti-causal, so
        # that `t_a * p_b` below is exactly DOVE Eq (15)'s `T^{f(*)b}`:
        # a FORWARD joint feature modulated by a BACKWARD probability map.
        if branch_mode == "directional":
            dir_a, dir_b = "forward", "backward"
        else:
            dir_a, dir_b = "symmetric", "symmetric"

        self.branch_a = DepthwiseSeparableBranch(
            dim, kernel_sizes[0], activation=activation, dropout=dropout,
            direction=dir_a,
        )
        self.branch_b = DepthwiseSeparableBranch(
            dim, kernel_sizes[1], activation=activation, dropout=dropout,
            direction=dir_b,
        )

        # DOVE Eq (14), one per direction of the mining.
        self.probe_a = TwoLayerFC(
            dim, hidden_mult=probe_hidden_mult, out_activation=probe_activation
        )
        self.probe_b = TwoLayerFC(
            dim, hidden_mult=probe_hidden_mult, out_activation=probe_activation
        )

        # DOVE Eq (8) functional form, applied to the two joint features.
        self.gate = nn.Linear(dim, dim) if use_eq8_gate else None

        # DOVE Eq (16).
        self.decoder = ResidualMLP(
            dim, hidden_mult=mlp_hidden_mult, activation=mlp_activation, dropout=dropout
        )

        self.gsa = (
            GatedSelfAttention(dim, num_heads=gsa_heads)
            if use_gated_self_attention
            else None
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {tokens.shape[-1]}")

        h = zero_invalid(tokens, mask)
        if self.gsa is not None:
            h = self.gsa(h, mask=mask)
            h = zero_invalid(h, mask)

        # Two local-context branches in place of DTGA's bi-GRU directions.
        # In "directional" mode b_a is the causal (forward) view and b_b the
        # anti-causal (backward) view, so the roles below mirror Eq (12)-(15).
        b_a = self.branch_a(h, mask)                       # ~ H~^f
        b_b = self.branch_b(h, mask)                       # ~ H~^b

        # DOVE Eq (13): joint features = branch output + input residual.
        t_a = b_a + h
        t_b = b_b + h

        # DOVE Eq (14): each direction's probability map, from the OTHER branch.
        p_b = self.probe_b(b_b)
        p_a = self.probe_a(b_a)

        # DOVE Eq (15): cross-direction mining, then sum into T_c.
        t_ab = t_a * p_b
        t_ba = t_b * p_a
        t_c = t_ab + t_ba

        # DOVE Eq (8) functional form, applied here to the joint features.
        if self.gate is not None:
            t_c = torch.sigmoid(self.gate(t_a * t_b)) * t_c

        # DOVE Eq (16): residual MLP decoder.
        out = self.decoder(t_c)
        return zero_invalid(out, mask)
