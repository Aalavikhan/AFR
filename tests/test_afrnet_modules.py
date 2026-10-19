"""
tests/test_afrnet_modules.py
-----------------------------
Equation-level tests for ROAM-Med, GTGA, DDTP, OT-Align and the objective.

These are **real assertions**, unlike `tests/test_1..6_*.py`, which only print
(see docs/KNOWN_ISSUES.md AFR-012). Every numeric expectation is either
hand-computed or recomputed independently from the module's own weights, so a
test failing means the mathematics diverged — not merely that a shape changed.

Runs two ways:

    pytest -q tests/test_afrnet_modules.py          # if pytest is installed
    python -X utf8 tests/test_afrnet_modules.py     # standalone, exit 1 on failure

Needs only `torch` on CPU. No BiomedCLIP, no network, no dataset: the
integration tests use `StubBackbone` below.

NOTE: as of writing these have NOT been executed in this environment.
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.afrnet import AFRNet                                  # noqa: E402
from models.modules import (                                      # noqa: E402
    DDTP,
    GTGA,
    AFRNetObjective,
    BidirectionalTripletRankingLoss,
    GatedSelfAttention,
    InterModalGuidanceAttention,
    IntraModalFusionAttention,
    OTAlign,
    ROAMMed,
    SpatialPyramidPool,
    clip_contrastive_loss,
    masked_mean,
)

TOL = 1e-5


def _seed(n: int = 0) -> None:
    torch.manual_seed(n)


# ===========================================================================
# Gated Self-Attention — DOVE Eq (6)-(11)
# ===========================================================================

def test_gated_self_attention_matches_equations_7_to_11():
    """Recompute Eq (7)-(11) by hand from the layer's own weights."""
    _seed()
    dim, b, n = 8, 2, 5
    gsa = GatedSelfAttention(dim, num_heads=1).eval()
    x = torch.randn(b, n, dim)

    with torch.no_grad():
        got = gsa(x)

        # Eq (7)
        q = gsa.w_q(x)
        k = gsa.w_k(x)
        v = gsa.w_v(x)
        # Eq (8): element-wise product of Q and K -> one FC -> sigmoid
        gate = torch.sigmoid(gsa.w_a(q * k))
        # Eq (9), (10): gate Q and K only; V stays ungated
        q_g, k_g = gate * q, gate * k
        # Eq (11)
        attn = ((q_g @ k_g.transpose(-2, -1)) / math.sqrt(dim)).softmax(-1)
        want = attn @ v

    assert torch.allclose(got, want, atol=TOL), (got - want).abs().max()


def test_gated_self_attention_leaves_v_ungated():
    """Eq (9)-(10) gate Q and K. Gating V too would change the module."""
    _seed()
    gsa = GatedSelfAttention(8, num_heads=1).eval()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        q = gsa.w_q(x); k = gsa.w_k(x); v = gsa.w_v(x)
        gate = torch.sigmoid(gsa.w_a(q * k))
        wrong_attn = (((gate * q) @ (gate * k).transpose(-2, -1)) / math.sqrt(8)).softmax(-1)
        wrong = wrong_attn @ (gate * v)          # V incorrectly gated
        got = gsa(x)
    assert not torch.allclose(got, wrong, atol=1e-4)


def test_gated_self_attention_masks_padding():
    _seed()
    gsa = GatedSelfAttention(8, num_heads=2).eval()
    x = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False
    with torch.no_grad():
        out = gsa(x, mask=mask)
    assert torch.allclose(out[0, 4:], torch.zeros(2, 8), atol=TOL)
    assert torch.isfinite(out).all()


# ===========================================================================
# ROAM-Med / IFA — DOVE Eq (17)-(22)
# ===========================================================================

def test_ifa_matches_equations_17_to_22():
    """Recompute Eq (17)-(22) by hand; no prior, no mask."""
    _seed()
    dim, b, n_m, n_r = 8, 2, 4, 6
    ifa = IntraModalFusionAttention(dim, prior_mode="none").eval()
    f_m = torch.randn(b, n_m, dim)
    f_r = torch.randn(b, n_r, dim)

    with torch.no_grad():
        got, s_mr, _ = ifa(f_m, f_r)

        f_m_p = ifa.w_m(f_m)                                   # Eq (17)
        f_r_p = ifa.w_r(f_r)                                   # Eq (18)
        want_s = torch.sigmoid(f_m_p @ f_r_p.transpose(1, 2))  # Eq (19)
        f_rm = want_s @ f_r_p + f_m_p                          # Eq (20)
        f_mr_ = want_s.transpose(1, 2) @ f_m_p + f_r_p         # Eq (21)
        want = ifa.w_l(torch.cat([f_rm, f_mr_], dim=1))        # Eq (22)

    assert torch.allclose(s_mr, want_s, atol=TOL)
    assert torch.allclose(got, want, atol=TOL)
    assert got.shape == (b, n_m + n_r, dim), got.shape


def test_eq19_is_sigmoid_not_softmax():
    """S_MR must be an element-wise gate in (0,1), NOT a normalised attention.

    This is the single most likely place for the module to be silently
    replaced by generic softmax attention.
    """
    _seed()
    ifa = IntraModalFusionAttention(8, prior_mode="none").eval()
    with torch.no_grad():
        _, s_mr, _ = ifa(torch.randn(2, 4, 8), torch.randn(2, 6, 8))
    assert ((s_mr > 0) & (s_mr < 1)).all(), "S_MR must lie strictly in (0,1)"
    row_sums = s_mr.sum(dim=2)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        "rows of S_MR sum to 1 -> softmax was used where Eq (19) specifies sigmoid"
    )


def test_ifa_residuals_are_present_and_cross_wired():
    """Eq (20) adds F'_M back (the query side); Eq (21) adds F'_R back.

    Note the residuals cannot even be swapped: `F_{R+M}` is `[B,Nm,D]` and
    `F'_R` is `[B,Nr,D]`, so a swap is a shape error, not a silent bug. What
    this test guards is that the residual term exists at all.
    """
    _seed()
    dim = 8
    ifa = IntraModalFusionAttention(dim, prior_mode="none").eval()
    f_m = torch.randn(1, 3, dim)
    f_r = torch.randn(1, 5, dim)
    with torch.no_grad():
        f_m_p, f_r_p = ifa.w_m(f_m), ifa.w_r(f_r)
        s = torch.sigmoid(f_m_p @ f_r_p.transpose(1, 2))
        attended_only_m = s @ f_r_p                        # residual omitted
        attended_only_r = s.transpose(1, 2) @ f_m_p
        want_m = attended_only_m + f_m_p                   # Eq (20)
        want_r = attended_only_r + f_r_p                   # Eq (21)
        got = ifa(f_m, f_r)[0]
        want = ifa.w_l(torch.cat([want_m, want_r], dim=1))
        wrong = ifa.w_l(torch.cat([attended_only_m, attended_only_r], dim=1))
    assert torch.allclose(got, want, atol=TOL)
    assert not torch.allclose(got, wrong, atol=1e-4), "residual terms missing"


def test_ifa_region_mask_zeroes_invalid_regions():
    _seed()
    ifa = IntraModalFusionAttention(8, prior_mode="none").eval()
    f_m = torch.randn(2, 3, 8)
    f_r = torch.randn(2, 5, 8)
    rmask = torch.ones(2, 5, dtype=torch.bool)
    rmask[0, 3:] = False
    with torch.no_grad():
        out, s_mr, keep = ifa(f_m, f_r, region_mask=rmask)
    assert torch.allclose(s_mr[0, :, 3:], torch.zeros(3, 2), atol=TOL)
    assert torch.allclose(out[0, 3 + 3:], torch.zeros(2, 8), atol=TOL)
    assert keep is not None and keep.shape == (2, 8)


def test_ifa_prior_multiply_scales_scores():
    """A mean-1 normalised prior must modulate S_MR multiplicatively."""
    _seed()
    ifa = IntraModalFusionAttention(8, prior_mode="multiply", prior_normalise=False).eval()
    f_m, f_r = torch.randn(1, 2, 8), torch.randn(1, 4, 8)
    prior = torch.full((1, 4), 0.5)
    with torch.no_grad():
        _, s_with, _ = ifa(f_m, f_r, region_prior=prior)
        f_m_p, f_r_p = ifa.w_m(f_m), ifa.w_r(f_r)
        base = torch.sigmoid(f_m_p @ f_r_p.transpose(1, 2))
    assert torch.allclose(s_with, base * 0.5, atol=TOL)


# ===========================================================================
# ROAM-Med / IGA — DOVE Eq (23)-(28)
# ===========================================================================

def test_iga_matches_equations_23_to_28_and_is_batch_coupled():
    _seed()
    dim, b, n_r, n_c = 8, 3, 5, 7
    iga = InterModalGuidanceAttention(dim, mode="batch").eval()
    f_r = torch.randn(b, n_r, dim)
    f_g = torch.randn(b, n_c, dim)

    with torch.no_grad():
        got, s_rg = iga(f_r, f_g)
        e_r = f_r.mean(dim=1)                                  # Eq (23)
        e_g = f_g.mean(dim=1)
        f_r_p = iga.w_r(e_r)                                   # Eq (24)
        f_g_p = iga.w_g(e_g)                                   # Eq (25)
        want_s = torch.sigmoid(f_r_p @ f_g_p.transpose(0, 1))  # Eq (26)
        f_rg = want_s @ f_g_p + f_g_p                          # Eq (27)
        want = iga.head(f_rg)                                  # Eq (28)

    assert s_rg.shape == (b, b), f"Eq (26) must be [B,B], got {s_rg.shape}"
    assert torch.allclose(s_rg, want_s, atol=TOL)
    assert torch.allclose(got, want, atol=TOL)


def test_iga_batch_mode_output_depends_on_other_samples():
    """Documents WARN-ROAM-1: faithful IGA couples mini-batch members."""
    _seed()
    iga = InterModalGuidanceAttention(8, mode="batch").eval()
    f_r = torch.randn(3, 4, 8)
    f_g = torch.randn(3, 6, 8)
    with torch.no_grad():
        out_a, _ = iga(f_r, f_g)
        f_g2 = f_g.clone()
        f_g2[2] = torch.randn(6, 8)          # perturb only sample 2
        out_b, _ = iga(f_r, f_g2)
    assert not torch.allclose(out_a[0], out_b[0], atol=1e-5), (
        "sample 0 did not change when sample 2 changed -> batch coupling lost"
    )


def test_iga_diagonal_mode_is_batch_independent():
    """The documented MOD-ROAM-1 alternative removes the coupling."""
    _seed()
    iga = InterModalGuidanceAttention(8, mode="diagonal").eval()
    f_r = torch.randn(3, 4, 8)
    f_g = torch.randn(3, 6, 8)
    with torch.no_grad():
        full, _ = iga(f_r, f_g)
        single, _ = iga(f_r[:1], f_g[:1])
    assert torch.allclose(full[0], single[0], atol=TOL)


def test_iga_masked_mean_ignores_padding():
    """Garbage in padded text slots must not reach Eq (23).

    `f_r` is built ONCE and reused: drawing it separately per call would
    advance the RNG and make the two calls differ for an unrelated reason.
    """
    _seed()
    iga = InterModalGuidanceAttention(8, mode="diagonal").eval()
    f_r = torch.randn(1, 3, 8)
    f_g = torch.randn(1, 6, 8)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 4:] = False
    polluted = f_g.clone()
    polluted[0, 4:] = 999.0
    with torch.no_grad():
        a, _ = iga(f_r, f_g, text_mask=mask)
        b_, _ = iga(f_r, polluted, text_mask=mask)
    assert torch.allclose(a, b_, atol=1e-4), "padding leaked into Eq (23)"


# ===========================================================================
# GTGA
# ===========================================================================

def test_gtga_branch_is_actually_depthwise():
    """`groups == channels` is what makes the convolution depthwise."""
    g = GTGA(dim=16, kernel_sizes=(3, 5))
    assert g.branch_a.depthwise.groups == 16
    assert g.branch_b.depthwise.groups == 16
    assert g.branch_a.depthwise.kernel_size == (3,)
    assert g.branch_b.depthwise.kernel_size == (5,)
    # depthwise weight is [C, 1, k], not [C, C, k]
    assert tuple(g.branch_a.depthwise.weight.shape) == (16, 1, 3)
    # pointwise mixes channels at one position
    assert g.branch_a.pointwise.kernel_size == (1,)
    assert g.branch_a.pointwise.groups == 1


def test_gtga_preserves_shape_and_zeroes_padding():
    _seed()
    g = GTGA(dim=16, kernel_sizes=(3, 5)).eval()
    x = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[0, 7:] = False
    with torch.no_grad():
        out = g(x, mask=mask)
    assert out.shape == x.shape
    assert torch.allclose(out[0, 7:], torch.zeros(5, 16), atol=TOL)
    assert torch.isfinite(out).all()


def test_gtga_padding_does_not_leak_into_valid_positions():
    """Garbage in padded slots must not change any valid output position.

    A depthwise kernel of width 5 reaches 2 neighbours, so without the
    zero_invalid calls the last valid tokens would be contaminated.
    """
    _seed()
    g = GTGA(dim=16, kernel_sizes=(3, 5)).eval()
    x = torch.randn(1, 12, 16)
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[0, 8:] = False

    polluted = x.clone()
    polluted[0, 8:] = 1e3

    with torch.no_grad():
        a = g(x, mask=mask)
        b_ = g(polluted, mask=mask)
    assert torch.allclose(a[0, :8], b_[0, :8], atol=1e-4), (
        "padded values changed valid outputs -> masking is broken"
    )


def test_gtga_variable_lengths_are_independent():
    _seed()
    g = GTGA(dim=16).eval()
    x = torch.randn(3, 10, 16)
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[1, 5:] = False
    mask[2, 2:] = False
    with torch.no_grad():
        out = g(x, mask=mask)
    assert torch.allclose(out[1, 5:], torch.zeros(5, 16), atol=TOL)
    assert torch.allclose(out[2, 2:], torch.zeros(8, 16), atol=TOL)


def test_gtga_directional_branches_have_opposite_receptive_fields():
    """The point of branch_mode='directional': A is causal, B anti-causal.

    Perturb the LAST token only.
      * causal branch  (i sees <= i): every earlier output must be UNCHANGED.
      * anti-causal    (i sees >= i): outputs within k-1 of the end MUST change.
    This is what makes the two views complementary, as DTGA's forward and
    backward GRU states are.
    """
    _seed()
    dim, k, length = 8, 3, 10
    g = GTGA(dim=dim, kernel_sizes=(k, k), branch_mode="directional").eval()
    assert g.branch_a.direction == "forward"
    assert g.branch_b.direction == "backward"

    x = torch.randn(1, length, dim)
    y = x.clone()
    y[0, length - 1] += 10.0                 # perturb only the final token

    with torch.no_grad():
        a_before, a_after = g.branch_a(x), g.branch_a(y)
        b_before, b_after = g.branch_b(x), g.branch_b(y)

    # causal: nothing before the perturbed position may move
    assert torch.allclose(a_before[0, : length - 1], a_after[0, : length - 1], atol=1e-6), (
        "causal branch leaked information backwards in time"
    )
    # anti-causal: positions within k-1 of the end must move
    assert not torch.allclose(
        b_before[0, length - k], b_after[0, length - k], atol=1e-4
    ), "anti-causal branch did not see the later token"


def test_gtga_directional_branches_differ_at_the_start_too():
    """Mirror check: perturb the FIRST token."""
    _seed()
    dim, k, length = 8, 3, 10
    g = GTGA(dim=dim, kernel_sizes=(k, k), branch_mode="directional").eval()
    x = torch.randn(1, length, dim)
    y = x.clone()
    y[0, 0] += 10.0

    with torch.no_grad():
        a_b, a_a = g.branch_a(x), g.branch_a(y)
        b_b, b_a = g.branch_b(x), g.branch_b(y)

    # causal: position k-1 sees token 0, so it must change
    assert not torch.allclose(a_b[0, k - 1], a_a[0, k - 1], atol=1e-4)
    # anti-causal: positions from k onwards cannot see token 0
    assert torch.allclose(b_b[0, k:], b_a[0, k:], atol=1e-6)


def test_gtga_symmetric_branch_sees_both_sides():
    """Contrast case: the default mode is NOT directional."""
    _seed()
    dim, k, length = 8, 3, 10
    g = GTGA(dim=dim, kernel_sizes=(k, k), branch_mode="symmetric").eval()
    assert g.branch_a.direction == "symmetric"
    x = torch.randn(1, length, dim)
    y = x.clone()
    y[0, 5] += 10.0
    with torch.no_grad():
        before, after = g.branch_a(x), g.branch_a(y)
    # a symmetric k=3 kernel at position 4 sees token 5 (to its right)
    assert not torch.allclose(before[0, 4], after[0, 4], atol=1e-4)
    # and at position 6 sees token 5 (to its left)
    assert not torch.allclose(before[0, 6], after[0, 6], atol=1e-4)


def test_gtga_all_branch_modes_preserve_length_and_masking():
    _seed()
    for mode in ("symmetric", "directional"):
        for kernels in ((3, 3), (3, 5), (5, 5)):
            g = GTGA(dim=8, kernel_sizes=kernels, branch_mode=mode).eval()
            x = torch.randn(2, 11, 8)
            mask = torch.ones(2, 11, dtype=torch.bool)
            mask[0, 7:] = False
            with torch.no_grad():
                out = g(x, mask=mask)
            assert out.shape == x.shape, (mode, kernels, out.shape)
            assert torch.allclose(out[0, 7:], torch.zeros(4, 8), atol=TOL), (mode, kernels)
            assert torch.isfinite(out).all()


def test_gtga_directional_padding_does_not_leak():
    """Masking must still hold in directional mode."""
    _seed()
    g = GTGA(dim=8, kernel_sizes=(5, 5), branch_mode="directional").eval()
    x = torch.randn(1, 12, 8)
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[0, 8:] = False
    polluted = x.clone()
    polluted[0, 8:] = 1e3
    with torch.no_grad():
        a, b_ = g(x, mask=mask), g(polluted, mask=mask)
    assert torch.allclose(a[0, :8], b_[0, :8], atol=1e-4)


def test_gtga_rejects_bad_branch_mode():
    try:
        GTGA(dim=8, branch_mode="bidirectional")
        raise AssertionError("unknown branch_mode should be rejected")
    except ValueError:
        pass


def test_gtga_rejects_even_kernels_and_wrong_branch_count():
    try:
        GTGA(dim=8, kernel_sizes=(2, 4))
        raise AssertionError("even kernel should be rejected")
    except ValueError:
        pass
    try:
        GTGA(dim=8, kernel_sizes=(3, 5, 7))
        raise AssertionError("3 branches should be rejected")
    except ValueError:
        pass


# ===========================================================================
# DDTP
# ===========================================================================

def test_ddtp_selects_98_of_196():
    """The user's intended configuration: 196 eligible patches, 50% -> 98."""
    d = DDTP(dim=32, ratio=0.5)
    assert d.resolve_k(196) == 98
    _seed()
    out = d(torch.randn(2, 196, 32), torch.randn(2, 32))
    assert out.features.shape == (2, 98, 32)
    assert out.indices.shape == (2, 98)
    assert out.all_scores.shape == (2, 196)


def test_ddtp_configurable_counts():
    assert DDTP(dim=8, ratio=0.25).resolve_k(196) == 49
    assert DDTP(dim=8, num_keep=10).resolve_k(196) == 10
    assert DDTP(dim=8, ratio=0.5).resolve_k(7) == 4        # round(3.5) -> 4
    assert DDTP(dim=8, ratio=0.5).resolve_k(1) == 1        # clamped to >= 1
    assert DDTP(dim=8, num_keep=500).resolve_k(196) == 196  # clamped to Np


def test_ddtp_indices_are_ascending_and_unique():
    _seed()
    out = DDTP(dim=16, ratio=0.5)(torch.randn(4, 20, 16), torch.randn(4, 16))
    idx = out.indices
    assert (idx[:, 1:] > idx[:, :-1]).all(), "indices must be strictly ascending"
    for b in range(idx.shape[0]):
        assert len(set(idx[b].tolist())) == idx.shape[1]


def test_ddtp_preserves_original_patch_identity():
    """Selected features must be the original patches at the kept indices."""
    _seed()
    d = DDTP(dim=16, ratio=0.5, score_weighting="none")
    patches = torch.randn(2, 20, 16)
    out = d(patches, torch.randn(2, 16))
    for b in range(2):
        gathered = patches[b, out.indices[b]]
        assert torch.allclose(out.features[b], gathered, atol=TOL)


def test_ddtp_picks_the_highest_scoring_patches():
    """Rank order must follow the relevance function, not arbitrary order."""
    _seed()
    d = DDTP(dim=16, ratio=0.5, score_weighting="none")
    patches = torch.randn(1, 10, 16)
    text = torch.randn(1, 16)
    out = d(patches, text)
    expected = out.all_scores[0].topk(5).indices.sort().values
    assert torch.equal(out.indices[0], expected)


def test_ddtp_never_selects_masked_patches_when_enough_remain():
    _seed()
    d = DDTP(dim=16, num_keep=4, score_weighting="none")
    patches = torch.randn(1, 12, 16)
    pmask = torch.ones(1, 12, dtype=torch.bool)
    pmask[0, :6] = False                       # only 6 eligible, need 4
    out = d(patches, torch.randn(1, 16), patch_mask=pmask)
    assert (out.indices >= 6).all(), out.indices
    assert out.mask.all()


def test_ddtp_flags_slots_when_valid_patches_are_too_few():
    _seed()
    d = DDTP(dim=16, num_keep=8, score_weighting="none")
    pmask = torch.zeros(1, 12, dtype=torch.bool)
    pmask[0, :3] = True                        # only 3 eligible, need 8
    out = d(torch.randn(1, 12, 16), torch.randn(1, 16), patch_mask=pmask)
    assert out.mask.sum().item() == 3, out.mask
    # invalid slots must be zeroed, not silently passed on as regions
    assert torch.allclose(out.features[0][~out.mask[0]],
                          torch.zeros(5, 16), atol=TOL)


def test_ddtp_gradient_reaches_the_scorer_but_not_the_indices():
    """Gradient flows through the score WEIGHTING; top-k indices are discrete."""
    _seed()
    d = DDTP(dim=16, ratio=0.5, score_weighting="softmax_norm")
    patches = torch.randn(2, 20, 16, requires_grad=True)
    text = torch.randn(2, 16, requires_grad=True)
    out = d(patches, text)
    out.features.sum().backward()

    assert d.patch_proj.weight.grad is not None
    assert d.patch_proj.weight.grad.abs().sum() > 0, "scorer received no gradient"
    assert d.text_proj.weight.grad.abs().sum() > 0
    assert patches.grad is not None and patches.grad.abs().sum() > 0
    assert out.indices.dtype == torch.int64
    assert not out.indices.is_floating_point()   # discrete: no gradient possible


def test_ddtp_score_weighting_none_starves_the_scorer():
    """Documents why 'none' is ablation-only."""
    _seed()
    d = DDTP(dim=16, ratio=0.5, score_weighting="none")
    out = d(torch.randn(2, 20, 16), torch.randn(2, 16))
    out.features.sum().backward()
    g = d.patch_proj.weight.grad
    assert g is None or g.abs().sum() == 0


def test_ddtp_softmax_norm_preserves_feature_scale():
    """Weights sum to k, so the mean weight is 1."""
    _seed()
    d = DDTP(dim=16, ratio=0.5, score_weighting="softmax_norm")
    out = d(torch.randn(3, 20, 16), torch.randn(3, 16))
    w = torch.softmax(out.scores, dim=1) * out.scores.shape[1]
    assert torch.allclose(w.sum(dim=1), torch.full((3,), 10.0), atol=1e-4)


# ===========================================================================
# SpatialPyramidPool (ROAM input adaptation)
# ===========================================================================

def test_pyramid_token_count_and_grid_requirement():
    p = SpatialPyramidPool(scales=(1, 2, 3, 7))
    assert p.num_tokens == 1 + 4 + 9 + 49 == 63
    out = p(torch.randn(2, 196, 32))
    assert out.shape == (2, 63, 32)
    # scale 1 is the global mean of all patches
    x = torch.randn(1, 196, 8)
    assert torch.allclose(SpatialPyramidPool(scales=(1,))(x)[0, 0],
                          x.mean(dim=1)[0], atol=1e-5)
    try:
        p(torch.randn(1, 195, 32))
        raise AssertionError("non-square patch count should be rejected")
    except ValueError:
        pass


# ===========================================================================
# OT-Align
# ===========================================================================

def test_ot_plan_satisfies_marginal_constraints():
    """P1 = a and P^T1 = b within tolerance — the defining OT constraints."""
    _seed()
    ot = OTAlign(dim=16, eps=0.05, max_iters=300, tol=1e-9)
    out = ot(torch.randn(3, 10, 16), torch.randn(3, 12, 16))
    row = out.plan.sum(dim=2)
    col = out.plan.sum(dim=1)
    assert torch.allclose(row, torch.full_like(row, 1 / 10), atol=1e-4), row
    assert torch.allclose(col, torch.full_like(col, 1 / 12), atol=1e-4), col
    assert out.max_marginal_error.item() < 1e-4
    assert (out.plan >= 0).all()
    assert torch.allclose(out.plan.sum(dim=(1, 2)), torch.ones(3), atol=1e-4)


def test_ot_identical_features_give_near_zero_cost():
    """Matched features -> cosine cost 0 on the diagonal -> loss ~ 0."""
    _seed()
    feats = F.normalize(torch.randn(1, 8, 16), dim=-1)
    ot = OTAlign(dim=16, eps=0.01, max_iters=500, tol=1e-10)
    out = ot(feats, feats.clone())
    assert out.loss.item() < 0.05, out.loss.item()


def test_ot_excludes_padded_tokens_from_the_plan():
    _seed()
    ot = OTAlign(dim=16, eps=0.05, max_iters=300)
    tokens = torch.randn(2, 12, 16)
    tmask = torch.ones(2, 12, dtype=torch.bool)
    tmask[0, 6:] = False
    out = ot(torch.randn(2, 10, 16), tokens, token_mask=tmask)
    assert torch.allclose(out.plan[0, :, 6:], torch.zeros(10, 6), atol=1e-6), (
        "padded tokens received transport mass"
    )
    col = out.plan[0].sum(dim=0)[:6]
    assert torch.allclose(col, torch.full_like(col, 1 / 6), atol=1e-4)


def test_ot_handles_variable_lengths_and_stays_finite():
    _seed()
    ot = OTAlign(dim=16)
    pmask = torch.ones(3, 10, dtype=torch.bool); pmask[2, 7:] = False
    tmask = torch.ones(3, 9, dtype=torch.bool);  tmask[0, 3:] = False
    out = ot(torch.randn(3, 10, 16), torch.randn(3, 9, 16),
             patch_mask=pmask, token_mask=tmask)
    assert torch.isfinite(out.loss) and torch.isfinite(out.plan).all()
    assert torch.isfinite(out.per_sample).all()


def test_ot_gradient_flows_into_features():
    _seed()
    ot = OTAlign(dim=16, detach_plan=True)
    patches = torch.randn(2, 10, 16, requires_grad=True)
    tokens = torch.randn(2, 9, 16, requires_grad=True)
    ot(patches, tokens).loss.backward()
    assert patches.grad is not None and patches.grad.abs().sum() > 0
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0


def test_ot_cost_matrix_is_cosine_distance():
    ot = OTAlign(dim=4, cost="cosine")
    x = torch.tensor([[[1.0, 0, 0, 0]]])
    y = torch.tensor([[[1.0, 0, 0, 0], [0.0, 1, 0, 0], [-1.0, 0, 0, 0]]])
    c = ot.cost_matrix(x, y)[0, 0]
    assert torch.allclose(c, torch.tensor([0.0, 1.0, 2.0]), atol=1e-5), c


def test_ot_is_not_a_cosine_similarity_loss():
    """Guard against the module being replaced by a plain similarity loss."""
    _seed()
    ot = OTAlign(dim=16, eps=0.05, max_iters=300)
    x, y = torch.randn(1, 10, 16), torch.randn(1, 12, 16)
    out = ot(x, y)
    naive = (1 - F.normalize(x, dim=-1).mean(1) @ F.normalize(y, dim=-1).mean(1).T).mean()
    assert not torch.allclose(out.loss, naive, atol=1e-3)
    # a real transport plan is sparse-ish, not uniform
    uniform = torch.full_like(out.plan, 1 / (10 * 12))
    assert not torch.allclose(out.plan, uniform, atol=1e-4)


# ===========================================================================
# Losses — DOVE Eq (30), (31)
# ===========================================================================

def test_triplet_loss_hand_computed():
    """Orthonormal pairs, margin 1.5, B=2 -> loss exactly 1.0.

    sim = I, pos = 1 for both. Off-diagonal cost = 1.5 - 1 + 0 = 0.5.
    i2t contributes 2 * 0.5, t2i contributes 2 * 0.5, total 2.0, /B = 1.0.
    """
    loss = BidirectionalTripletRankingLoss(margin=1.5, reduction="mean")
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    got = loss(v, v.clone())
    assert abs(got.item() - 1.0) < 1e-6, got.item()


def test_triplet_loss_zero_when_margin_satisfied():
    loss = BidirectionalTripletRankingLoss(margin=0.2)
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert loss(v, v.clone()).item() == 0.0


def test_triplet_loss_excludes_the_positive_pair():
    loss = BidirectionalTripletRankingLoss(margin=2.0, reduction="sum")
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # every off-diagonal term is 2.0 - 1 + 0 = 1.0; 2 directions x 2 anchors
    assert abs(loss(v, v.clone()).item() - 4.0) < 1e-6


def test_triplet_loss_single_sample_has_no_negatives():
    loss = BidirectionalTripletRankingLoss(margin=0.2)
    assert loss(torch.randn(1, 8), torch.randn(1, 8)).item() == 0.0


def test_clip_loss_matches_baseline_formula():
    """`clip_contrastive_loss` must equal `models/baseline.py::clip_loss`."""
    _seed()
    img = F.normalize(torch.randn(4, 8), dim=-1)
    txt = F.normalize(torch.randn(4, 8), dim=-1)
    logit_scale = torch.tensor(2.0).log()

    got = clip_contrastive_loss(img, txt, logit_scale)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * img @ txt.T
    labels = torch.arange(4)
    want = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    assert torch.allclose(got, want, atol=TOL)


def test_objective_reports_every_term_and_respects_zero_weights():
    _seed()
    v, t = torch.randn(4, 8), torch.randn(4, 8)
    ls = torch.tensor(1.0).log()

    obj = AFRNetObjective(primary="clip", lambda_ot=0.0, lambda_g=0.0)
    out = obj(v, t, logit_scale=ls)
    assert set(out.parts) == {"primary", "total"}
    assert torch.allclose(out.total, out.parts["primary"])

    obj2 = AFRNetObjective(primary="clip", lambda_ot=0.5, lambda_g=2.0)
    ot = torch.tensor(3.0)
    out2 = obj2(v, t, logit_scale=ls, ot_loss=ot,
                v_global=torch.randn(4, 8), t_global=torch.randn(4, 8))
    assert set(out2.parts) == {"primary", "ot", "global_constraint", "total"}
    expected = (out2.parts["primary"] + 0.5 * ot
                + 2.0 * out2.parts["global_constraint"])
    assert torch.allclose(out2.total, expected, atol=TOL)


def test_objective_errors_when_a_weighted_term_is_missing():
    obj = AFRNetObjective(primary="clip", lambda_ot=1.0)
    try:
        obj(torch.randn(2, 8), torch.randn(2, 8), logit_scale=torch.tensor(0.0))
        raise AssertionError("missing ot_loss should raise")
    except ValueError:
        pass


# ===========================================================================
# Integration — AFRNet with a stub backbone (no BiomedCLIP, no network)
# ===========================================================================

class StubBackbone(torch.nn.Module):
    """Stands in for BiomedCLIPBaseline: same interface, trivial internals.

    Deliberately holds parameters so the frozen-backbone tests are meaningful.
    """

    def __init__(self, dim: int = 768, num_patches: int = 196, vocab: int = 50):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.embed = torch.nn.Embedding(vocab, dim)
        self.visual_proj = torch.nn.Linear(dim, dim)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self._frozen = False

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def pad_token_id(self) -> int:
        return 0

    def text_pad_mask(self, tokens):
        return tokens != self.pad_token_id()

    def get_patch_tokens(self, images):
        b = images.shape[0]
        flat = images.flatten(1)[:, : self.dim]
        if flat.shape[1] < self.dim:
            flat = F.pad(flat, (0, self.dim - flat.shape[1]))
        base = self.visual_proj(flat)
        return base.unsqueeze(1).expand(b, self.num_patches, self.dim) \
            + torch.linspace(0, 1, self.num_patches, device=images.device)[None, :, None]

    def encode_word_tokens(self, tokens):
        return self.embed(tokens), self.text_pad_mask(tokens)


def _build(**kw):
    torch.manual_seed(0)
    backbone = StubBackbone(dim=64)
    model = AFRNet(backbone, dim=64, **kw)
    return backbone, model


def test_afrnet_forward_shapes_and_196_to_98_flow():
    backbone, model = _build()
    assert model.num_regions == 98
    assert model.num_scales == 63
    images = torch.randn(2, 3, 8, 8)
    tokens = torch.randint(1, 50, (2, 20))
    tokens[0, 15:] = 0                       # padding
    out = model(images, tokens)

    assert out.v_final.shape == (2, 64)
    assert out.t_final.shape == (2, 64)
    assert out.v_global.shape == (2, 64)
    assert out.t_global.shape == (2, 64)
    assert out.aux["f_r"].shape == (2, 98, 64)
    assert out.aux["f_m"].shape == (2, 63, 64)
    assert out.aux["f_mr"].shape == (2, 63 + 98, 64)
    assert out.aux["s_mr"].shape == (2, 63, 98)
    assert out.aux["s_rg"].shape == (2, 2)
    for key in ("v_final", "t_final", "v_global", "t_global"):
        assert torch.isfinite(getattr(out, key)).all(), key
    assert out.ot_loss is not None and torch.isfinite(out.ot_loss)


def test_afrnet_runs_without_any_segmentation_input():
    """Segmentation is optional: no masks, no prior, no fabricated data."""
    _, model = _build()
    out = model(torch.randn(2, 3, 8, 8), torch.randint(1, 50, (2, 12)))
    assert torch.isfinite(out.v_final).all()
    assert out.aux["region_prior"] is not None    # falls back to DDTP scores


def test_afrnet_accepts_external_segmentation_interfaces():
    _, model = _build()
    images = torch.randn(2, 3, 8, 8)
    tokens = torch.randint(1, 50, (2, 12))
    pmask = torch.ones(2, 196, dtype=torch.bool)
    pmask[:, 150:] = False
    prior = torch.rand(2, 196)
    out = model(images, tokens, patch_mask=pmask, region_prior_external=prior)
    assert (out.aux["ddtp_indices"] < 150).all()
    assert out.aux["region_prior"].shape == (2, 98)


def test_afrnet_backbone_is_frozen_and_gets_no_gradients():
    backbone, model = _build()
    assert all(not p.requires_grad for p in backbone.parameters())

    out = model(torch.randn(2, 3, 8, 8), torch.randint(1, 50, (2, 12)))
    obj = AFRNetObjective(primary="clip", lambda_ot=1.0, lambda_g=1.0)
    loss = obj(out.v_final, out.t_final, logit_scale=backbone.logit_scale,
               ot_loss=out.ot_loss, v_global=out.v_global,
               t_global=out.t_global).total
    loss.backward()

    for name, p in backbone.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"backbone {name} got gradient"


def test_afrnet_all_four_modules_receive_gradients():
    backbone, model = _build()
    out = model(torch.randn(3, 3, 8, 8), torch.randint(1, 50, (3, 14)))
    obj = AFRNetObjective(primary="clip", lambda_ot=1.0)
    obj(out.v_final, out.t_final, logit_scale=backbone.logit_scale,
        ot_loss=out.ot_loss).total.backward()

    for prefix in ("gtga", "ddtp", "roam"):
        grads = [
            p.grad.abs().sum().item()
            for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        ]
        assert grads and sum(grads) > 0, f"{prefix} received no gradient"


def test_afrnet_optimizer_sees_only_module_parameters():
    backbone, model = _build()
    trainable = model.trainable_module_names()
    assert trainable, "nothing is trainable"
    assert not any(n.startswith("backbone.") for n in trainable), trainable
    assert any(n.startswith("gtga.") for n in trainable)
    assert any(n.startswith("ddtp.") for n in trainable)
    assert any(n.startswith("roam.") for n in trainable)


def test_afrnet_rejects_independent_encoding():
    """Cross-conditioned embeddings cannot be produced one modality at a time."""
    _, model = _build()
    for fn, arg in ((model.encode_image, torch.randn(1, 3, 8, 8)),
                    (model.encode_text, torch.randint(1, 50, (1, 8)))):
        try:
            fn(arg)
            raise AssertionError("expected NotImplementedError")
        except NotImplementedError:
            pass


def test_afrnet_diagonal_iga_is_batch_size_invariant():
    _, model = _build(roam_iga_mode="diagonal", ot_enabled=False)
    model.eval()
    images = torch.randn(3, 3, 8, 8)
    tokens = torch.randint(1, 50, (3, 12))
    with torch.no_grad():
        full = model(images, tokens)
        one = model(images[:1], tokens[:1])
    assert torch.allclose(full.t_final[0], one.t_final[0], atol=1e-4)


def test_masked_mean_matches_plain_mean_without_a_mask():
    _seed()
    x = torch.randn(2, 5, 4)
    assert torch.allclose(masked_mean(x, None), x.mean(dim=1), atol=TOL)
    m = torch.ones(2, 5, dtype=torch.bool)
    m[0, 3:] = False
    assert torch.allclose(masked_mean(x, m)[0], x[0, :3].mean(dim=0), atol=TOL)


def test_masked_mean_all_invalid_is_zero_not_nan():
    x = torch.randn(1, 4, 3)
    m = torch.zeros(1, 4, dtype=torch.bool)
    got = masked_mean(x, m)
    assert torch.isfinite(got).all() and torch.allclose(got, torch.zeros(1, 3))


# ===========================================================================
# Segmentor separation (regression guard)
# ===========================================================================

def test_no_module_references_a_segmentor():
    """The segmentor must stay external. Nothing here may import or call it."""
    root = pathlib.Path(__file__).resolve().parent.parent
    targets = list((root / "models").rglob("*.py"))
    assert targets
    banned = ("import segment", "from segment", "Segmentor(", "segmentor.")
    for path in targets:
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token.lower() not in text, f"{path} references a segmentor: {token}"


# ===========================================================================
# Standalone runner
# ===========================================================================

def _main() -> int:
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures: list[str] = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures.append(name)
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("failed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
