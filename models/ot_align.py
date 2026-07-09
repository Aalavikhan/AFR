"""
models/ot_align.py
------------------
Module 4 — OT-Align (auxiliary fine-grained alignment loss).

Ports DFIM's FSM: cost C = 1 - cos_sim between retained patches and
phrase-enhanced words; entropy-regularised OT via Sinkhorn-Knopp; loss = <P*, C>.
Also returns P* for visualisation (column j = which patches explain word j).

Implementation notes
---------------------
* Batched, log-domain Sinkhorn written in pure torch -> differentiable, GPU-safe,
  no per-sample POT loop.
* Forced fp32 inside the solver (plan risk item: Sinkhorn is unstable in bf16).
* PAD words are handled by zero target-marginal (log -inf) so no mass is routed
  to padding, and their cost column is irrelevant.

Shapes:
    V_tilde : [B, M, 768]   F_G : [B, L, 768]   text_mask : [B, L] (True = valid)
    -> L_OT : scalar        P* : [B, M, L]
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_log(
    C: torch.Tensor,        # [B, M, N] cost (>= 0)
    a_log: torch.Tensor,    # [B, M] log source marginal
    b_log: torch.Tensor,    # [B, N] log target marginal (may contain -inf for PAD)
    eps: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    """Return transport plan P [B, M, N]. Runs in the caller's dtype (use fp32)."""
    K = -C / eps                                          # [B, M, N]
    f = torch.zeros_like(a_log)                           # [B, M]
    g = torch.zeros_like(b_log)                           # [B, N]
    for _ in range(n_iters):
        f = a_log - torch.logsumexp(K + g.unsqueeze(1), dim=2)
        g = b_log - torch.logsumexp(K + f.unsqueeze(2), dim=1)
    P = torch.exp(f.unsqueeze(2) + K + g.unsqueeze(1))    # [B, M, N]
    return P


class OTAlign(nn.Module):
    def __init__(self, eps: float = 0.05, n_iters: int = 50):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters

    def forward(
        self,
        V_tilde: torch.Tensor,                     # [B, M, D] (L2-normalised)
        F_G: torch.Tensor,                         # [B, L, D]
        text_mask: Optional[torch.Tensor] = None,  # [B, L] True = valid token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, D = V_tilde.shape
        L = F_G.shape[1]

        # Force fp32 for numerical stability (plan §8.8 / risk register).
        v = V_tilde.float()
        w = F.normalize(F_G.float(), dim=-1)

        # Cosine cost in [0, 2]
        C = 1.0 - torch.bmm(v, w.transpose(1, 2))          # [B, M, N]

        # Source marginal: uniform over M patches
        a_log = torch.full((B, M), -torch.log(torch.tensor(float(M))), device=v.device)

        # Target marginal: uniform over VALID tokens (PAD -> -inf)
        if text_mask is None:
            text_mask = torch.ones(B, L, dtype=torch.bool, device=v.device)
        valid = text_mask.float()                          # [B, L]
        counts = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        b = valid / counts                                 # normalised per row
        b_log = torch.log(b.clamp(min=1e-12))
        b_log = b_log.masked_fill(~text_mask, float("-inf"))

        # Prevent -inf * cost interactions: park PAD columns at high cost
        C = C.masked_fill(~text_mask.unsqueeze(1), 1e4)

        P = sinkhorn_log(C, a_log, b_log, eps=self.eps, n_iters=self.n_iters)

        # Loss = <P*, C> over valid entries, averaged over batch
        L_OT = (P * C.clamp(max=2.0)).sum(dim=(1, 2)).mean()
        return L_OT, P
