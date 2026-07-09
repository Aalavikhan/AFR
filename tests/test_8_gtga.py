"""TEST 8 — GTGA shape + gradient + pad-mask (synthetic)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
from models.gtga import GTGA

B, L, D = 4, 256, 768
T_RG = torch.randn(B, L, D, requires_grad=True)
mask = torch.zeros(B, L, dtype=torch.bool)
mask[:, 200:] = True   # pretend tokens 200..255 are PAD

gtga = GTGA(dim=D, num_heads=8)
F_G = gtga(T_RG, key_padding_mask=mask)
assert F_G.shape == (B, L, D), F_G.shape

F_G.pow(2).mean().backward()
assert T_RG.grad is not None and T_RG.grad.abs().sum() > 0
n = sum(1 for p in gtga.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
tot = sum(1 for _ in gtga.parameters())
print(f"GTGA params receiving grad: {n}/{tot}")
print("PASS — GTGA shapes + gradients OK")
