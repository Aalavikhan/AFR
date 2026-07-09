"""TEST 9 — DDTP retention count, mask correctness, gradient flow (synthetic)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
from models.ddtp import DDTP

B, N, D = 4, 196, 768
V_MR = torch.randn(B, N, D, requires_grad=True)
w_eos = torch.randn(B, D)
v_cls = torch.randn(B, D)

ddtp = DDTP(dim=D, keep_ratio=0.5, num_patches=N)
v_final, V_tilde, mask, aux = ddtp(V_MR, w_eos, v_cls)

assert ddtp.M == 98, ddtp.M
assert v_final.shape == (B, D)
assert V_tilde.shape == (B, 98, D), V_tilde.shape
assert mask.dtype == torch.bool and mask.sum(dim=1).unique().tolist() == [98]
# retained features are L2-normalised
norms = V_tilde.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

(v_final.pow(2).mean() + V_tilde.pow(2).mean()).backward()
per_patch_grad = V_MR.grad.abs().sum(dim=-1)          # [B, N]
frac_with_grad = (per_patch_grad > 0).float().mean().item()

# The critical check: the SCORING network must receive gradient, or it is dead.
scorer_params = list(ddtp.cross_attn.parameters()) + [ddtp.prompts]
scorer_grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0 for p in scorer_params)

print(f"DDTP M = {ddtp.M}, retained-per-row = {mask.sum(1).unique().tolist()}")
print(f"fraction of patches receiving gradient: {frac_with_grad:.3f}  "
      f"(hard pruning -> ~keep_ratio, expected)")
print(f"scoring network (prompts + cross-attn) receives gradient: {scorer_grad_ok}")
assert scorer_grad_ok, "SCORER IS DEAD — prompts/cross-attn got no gradient"
# retained patches themselves must carry gradient
assert (per_patch_grad[mask] > 0).all(), "retained patches must have gradient"
print("PASS — DDTP retention + live scoring network OK")
