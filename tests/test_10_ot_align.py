"""TEST 10 — OT-Align: plan validity, gradient, PAD handling, alignment sanity."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
import torch.nn.functional as F
from models.ot_align import OTAlign

B, M, L, D = 4, 98, 256, 768
V_tilde = F.normalize(torch.randn(B, M, D), dim=-1).requires_grad_(True)
F_G = torch.randn(B, L, D, requires_grad=True)
mask = torch.ones(B, L, dtype=torch.bool); mask[:, 200:] = False

ot = OTAlign(eps=0.05, n_iters=50)
L_OT, P = ot(V_tilde, F_G, text_mask=mask)

assert P.shape == (B, M, L), P.shape
# source marginal ~ uniform 1/M; no mass on PAD columns
row_sums = P.sum(dim=2)                       # [B, M]  -> ~1/M each
pad_mass = P[:, :, 200:].sum().item()
assert abs(row_sums.sum(dim=1).mean().item() - 1.0) < 1e-2, row_sums.sum(1).mean()
assert pad_mass < 1e-3, f"mass leaked onto PAD: {pad_mass}"

L_OT.backward()
assert V_tilde.grad.abs().sum() > 0 and F_G.grad.abs().sum() > 0

# Sanity: an ALIGNED pair (words = subset of patches) should cost less than a
# random/shuffled pair. This is the plan's "random baseline for OT" check.
base = F.normalize(torch.randn(B, L, D), dim=-1)
Vt_aligned = F.normalize(base[:, :M, :] + 0.01 * torch.randn(B, M, D), dim=-1)
L_aligned, _ = ot(Vt_aligned, base)
L_random,  _ = ot(F.normalize(torch.randn(B, M, D), dim=-1), base)
print(f"L_OT (grad test)         : {L_OT.item():.4f}")
print(f"L_OT aligned pair        : {L_aligned.item():.4f}")
print(f"L_OT random pair         : {L_random.item():.4f}")
assert L_aligned.item() < L_random.item(), "aligned should cost less than random"
print("PASS — OT-Align plan valid, PAD-safe, gradient OK, aligned<random")
