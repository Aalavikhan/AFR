"""TEST 7 — ROAM-Med shape + gradient flow (synthetic, no backbone needed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
from models.roam import ROAMMed

B, N, L, D = 4, 196, 256, 768
V = torch.randn(B, N, D, requires_grad=True)
W = torch.randn(B, L, D, requires_grad=True)

roam = ROAMMed(dim=D, num_heads=8)

# Case A: no prior supplied -> backbone-derived
V_MR, T_RG, P = roam(V, W)
assert V_MR.shape == (B, N, D), V_MR.shape
assert T_RG.shape == (B, L, D), T_RG.shape
assert P.shape == (B, N) and P.min() >= 0 and P.max() <= 1.0 + 1e-5

# Case B: explicit prior [B,14,14]
prior = torch.rand(B, 14, 14)
V_MR2, T_RG2, P2 = roam(V, W, P_seg=prior)
assert V_MR2.shape == (B, N, D)

# gradient flow to both inputs and to module params
loss = V_MR.pow(2).mean() + T_RG.pow(2).mean()
loss.backward()
assert V.grad is not None and W.grad is not None
assert V.grad.abs().sum() > 0 and W.grad.abs().sum() > 0
n_params_with_grad = sum(1 for p in roam.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"ROAM params receiving grad: {n_params_with_grad}/{sum(1 for _ in roam.parameters())}")
print("prior range:", round(P.min().item(), 3), round(P.max().item(), 3))
print("PASS — ROAM-Med shapes + gradients OK")
