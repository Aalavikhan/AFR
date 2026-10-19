"""
models/modules/ot_align.py
---------------------------
OT-Align — optimal-transport alignment between visual patches and text tokens
(this project's **custom** module).

OT-Align has **no counterpart in DOVE**. The paper's objective is a
bidirectional triplet ranking loss plus a triplet-based global constraint
(Eq 30-31); it contains no transport problem, no Sinkhorn iteration and no
entropic regularisation. Nothing here may be attributed to the paper.

Formulation (chosen and documented by this repository)
------------------------------------------------------
For each sample, with `n` valid patches `x_i` and `m` valid tokens `y_j`:

    cost        C_ij = 1 - cos(x_i, y_j)              in [0, 2]
    marginals   a = uniform over valid patches (sum 1)
                b = uniform over valid tokens  (sum 1)
    problem     min_<P>  <P, C>  -  eps * H(P)
                s.t.     P 1 = a,  P^T 1 = b,  P >= 0
                where    H(P) = -sum_ij P_ij (log P_ij - 1)
    loss        L_OT = <P, C>        (transport cost of the optimal plan)

This is the standard **entropic-regularised Kantorovich problem** solved by
Sinkhorn-Knopp (Cuturi, 2013). Entropic regularisation is what makes the
problem strictly convex, cheap, and differentiable, so gradients flow into
whatever produced `x` and `y`.

Why a hand-written log-domain solver rather than POT
-----------------------------------------------------
`pot` is a declared dependency of this project but is **not used here**:

1. The solver must be differentiable in PyTorch and run on the training
   device; POT's `sinkhorn` returns a plan but round-tripping through it adds
   friction for autograd and for batched GPU tensors.
2. Log-domain stabilisation is required. The naive kernel `exp(-C/eps)`
   underflows to 0 for small `eps` in fp16/bf16, after which the scaling
   vectors become NaN. This implementation runs entirely in log space using
   `logsumexp`, which is stable for the `eps` values we care about.
3. Masking variable-length text per batch element needs `-inf` cost padding,
   which is natural here and awkward through POT's dense API.

Precision
---------
The solver **forces float32 internally** regardless of the surrounding
autocast dtype, then casts the scalar loss back. Sinkhorn under fp16/bf16 is
numerically unreliable (see WARN-OT-1 in docs/FIDELITY_AUDIT.md); the cost of
fp32 here is negligible against a ViT forward pass.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import as_bool_mask


class OTOutput(NamedTuple):
    loss: torch.Tensor          # scalar, batch-reduced transport cost
    per_sample: torch.Tensor    # [B] transport cost of each sample
    plan: torch.Tensor          # [B, n, m] transport plan P
    cost: torch.Tensor          # [B, n, m] cost matrix C
    iterations: int             # Sinkhorn iterations actually run
    max_marginal_error: torch.Tensor  # scalar, worst |P1 - a| or |P^T1 - b|


def log_sinkhorn(
    cost: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float = 0.05,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    """Entropic OT in the log domain. Returns `(log_plan, iterations_run)`.

    Args:
        cost:  [B, n, m] non-negative cost. Use `+inf` to forbid a pairing.
        log_a: [B, n] log of the source marginal (`-inf` for invalid rows).
        log_b: [B, m] log of the target marginal (`-inf` for invalid columns).
        eps:   entropic regularisation strength. Smaller = closer to exact OT
               but slower to converge and more prone to numerical trouble.
        max_iters: hard cap on iterations.
        tol:   stop when the max change in the dual potential `f` falls below
               this. Checked every iteration.

    The recursion is the standard log-domain Sinkhorn:
        f_i <- eps*log_a_i - eps*logsumexp_j((g_j - C_ij)/eps)
        g_j <- eps*log_b_j - eps*logsumexp_i((f_i - C_ij)/eps)
        log P_ij = (f_i + g_j - C_ij)/eps
    """
    # -C/eps with +inf cost -> -inf kernel, which logsumexp handles as 0 mass.
    neg_c = -cost / eps
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    iterations = 0
    for it in range(1, max_iters + 1):
        iterations = it
        f_prev = f

        # f update: logsumexp over j (columns)
        m_fg = neg_c + (g / eps).unsqueeze(1)            # [B, n, m]
        f = eps * (log_a - torch.logsumexp(m_fg, dim=2))

        # g update: logsumexp over i (rows)
        m_gf = neg_c + (f / eps).unsqueeze(2)            # [B, n, m]
        g = eps * (log_b - torch.logsumexp(m_gf, dim=1))

        # Invalid rows/columns carry -inf marginals; keep their potentials
        # finite so they cannot poison the logsumexp of valid entries.
        f = torch.nan_to_num(f, nan=0.0, neginf=0.0, posinf=0.0)
        g = torch.nan_to_num(g, nan=0.0, neginf=0.0, posinf=0.0)

        delta = (f - f_prev).abs().max()
        if torch.isfinite(delta) and delta.item() < tol:
            break

    log_plan = (f.unsqueeze(2) + g.unsqueeze(1) + neg_c)
    return log_plan, iterations


class OTAlign(nn.Module):
    """Optimal-transport alignment loss between patches and text tokens.

    Args:
        dim:        feature size of both inputs. `None` disables the shape check.
        eps:        entropic regularisation (default 0.05).
        max_iters:  Sinkhorn iteration cap (default 100).
        tol:        dual-potential convergence tolerance (default 1e-6).
        cost:       "cosine" -> `C = 1 - cos(x, y)`      (default)
                    "euclidean" -> `C = ||x_hat - y_hat||_2` on L2-normalised
                    features, which equals `sqrt(2 - 2cos)`.
        normalise:  L2-normalise features before computing the cost. Required
                    for "cosine" to be a true cosine; kept configurable so the
                    choice is explicit.
        project:    add a trainable linear map on each modality before the
                    cost. Off by default so the loss acts directly on the
                    module outputs.
        detach_plan: treat `P` as a constant when back-propagating, i.e. use
                    the envelope-theorem gradient `d/dtheta <P, C>` with `P`
                    fixed. This is cheaper and more stable than
                    differentiating through the whole iteration, and is exact
                    for the unregularised optimum. Default True.

    Shapes:
        patches:      [B, n, D]
        tokens:       [B, m, D]
        patch_mask:   [B, n] bool (optional)
        token_mask:   [B, m] bool (optional)
    """

    _BIG = 1.0e4   # finite stand-in for +inf cost; keeps fp32 logsumexp sane

    def __init__(
        self,
        dim: Optional[int] = 768,
        eps: float = 0.05,
        max_iters: int = 100,
        tol: float = 1e-6,
        cost: str = "cosine",
        normalise: bool = True,
        project: bool = False,
        detach_plan: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        if cost not in ("cosine", "euclidean"):
            raise ValueError(f"cost must be cosine/euclidean, got {cost!r}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.dim = dim
        self.eps = float(eps)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.cost_kind = cost
        self.normalise = normalise
        self.detach_plan = detach_plan
        self.reduction = reduction

        if project:
            if dim is None:
                raise ValueError("project=True requires an explicit dim")
            self.proj_v = nn.Linear(dim, dim)
            self.proj_t = nn.Linear(dim, dim)
        else:
            self.proj_v = None
            self.proj_t = None

    # ------------------------------------------------------------------
    def cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """[B, n, D], [B, m, D] -> [B, n, m] non-negative cost."""
        if self.normalise:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        sim = x @ y.transpose(1, 2)
        if self.cost_kind == "cosine":
            return (1.0 - sim).clamp_min(0.0)
        # euclidean on normalised features
        return (2.0 - 2.0 * sim).clamp_min(0.0).sqrt()

    # ------------------------------------------------------------------
    def forward(
        self,
        patches: torch.Tensor,
        tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> OTOutput:
        if self.dim is not None:
            if patches.shape[-1] != self.dim or tokens.shape[-1] != self.dim:
                raise ValueError(
                    f"OTAlign expects dim {self.dim}; got {patches.shape[-1]} "
                    f"and {tokens.shape[-1]}"
                )

        if self.proj_v is not None:
            patches = self.proj_v(patches)
            tokens = self.proj_t(tokens)

        # Sinkhorn in fp32 regardless of autocast. See module docstring.
        x = patches.float()
        y = tokens.float()

        b, n, _ = x.shape
        m = y.shape[1]

        pm = as_bool_mask(patch_mask, x) if patch_mask is not None else \
            torch.ones(b, n, dtype=torch.bool, device=x.device)
        tm = as_bool_mask(token_mask, y) if token_mask is not None else \
            torch.ones(b, m, dtype=torch.bool, device=y.device)

        cost = self.cost_matrix(x, y)                          # [B, n, m]

        # Forbid any pairing that touches an invalid patch or token.
        pair_valid = pm.unsqueeze(2) & tm.unsqueeze(1)          # [B, n, m]
        cost_solver = cost.masked_fill(~pair_valid, self._BIG)

        # Uniform marginals over VALID entries only.
        n_valid = pm.sum(dim=1).clamp_min(1).to(x.dtype)        # [B]
        m_valid = tm.sum(dim=1).clamp_min(1).to(y.dtype)
        log_a = torch.where(
            pm, -torch.log(n_valid).unsqueeze(1).expand(-1, n),
            torch.full_like(cost[:, :, 0], float("-inf")),
        )
        log_b = torch.where(
            tm, -torch.log(m_valid).unsqueeze(1).expand(-1, m),
            torch.full_like(cost[:, 0, :], float("-inf")),
        )

        solver_ctx = torch.no_grad() if self.detach_plan else _NullCtx()
        with solver_ctx:
            log_plan, iters = log_sinkhorn(
                cost_solver.detach() if self.detach_plan else cost_solver,
                log_a, log_b,
                eps=self.eps, max_iters=self.max_iters, tol=self.tol,
            )
            plan = log_plan.exp()
            plan = torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
            plan = plan * pair_valid.to(plan.dtype)

        if self.detach_plan:
            plan = plan.detach()

        # Transport cost of the plan. Gradient flows through `cost` (hence into
        # the features and any upstream trainable module) with `plan` fixed.
        per_sample = (plan * cost).sum(dim=(1, 2))              # [B]

        # Constraint check, reported so a test can assert it.
        row_err = (plan.sum(dim=2) - log_a.exp().nan_to_num(0.0)).abs()
        col_err = (plan.sum(dim=1) - log_b.exp().nan_to_num(0.0)).abs()
        max_marginal_error = torch.maximum(row_err.max(), col_err.max()).detach()

        if self.reduction == "mean":
            loss = per_sample.mean()
        elif self.reduction == "sum":
            loss = per_sample.sum()
        else:
            loss = per_sample

        return OTOutput(
            loss=loss.to(patches.dtype),
            per_sample=per_sample,
            plan=plan,
            cost=cost,
            iterations=iters,
            max_marginal_error=max_marginal_error,
        )


class _NullCtx:
    """`with` block that does nothing (keeps the solver call site uniform)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
