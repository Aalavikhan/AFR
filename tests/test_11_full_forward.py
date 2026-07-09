"""TEST 11 — AFR-Net full assembly forward/backward with a MOCK backbone.

The mock mimics BiomedCLIPBaseline's interface so we can verify wiring and
gradient flow through EVERY module without downloading real weights.
Replace MockBackbone with the real BiomedCLIPBaseline on your machine.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.afrnet import AFRNet
from losses.losses import AFRLoss, LossConfig

D_HID, D_PROJ, N, L = 768, 512, 196, 256


class MockBackbone(nn.Module):
    """Stand-in for BiomedCLIPBaseline (trainable so we can check grad flow)."""
    def __init__(self):
        super().__init__()
        self.embed_dim = D_PROJ
        self.patch = nn.Linear(3 * 16 * 16, D_HID)
        self.word = nn.Embedding(30000, D_HID)
        self.img_head = nn.Linear(D_HID, D_PROJ)
        self.txt_head = nn.Linear(D_HID, D_PROJ)

    def get_patch_tokens(self, images):            # images [B,3,224,224]
        B = images.shape[0]
        patches = images.unfold(2, 16, 16).unfold(3, 16, 16)       # [B,3,14,14,16,16]
        patches = patches.reshape(B, 3, N, 16 * 16).permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.patch(patches)                 # [B, N, D]

    def get_word_tokens(self, tokens):             # tokens [B,L] long
        return self.word(tokens)                   # [B, L, D]

    def encode_image(self, images):
        return F.normalize(self.img_head(self.get_patch_tokens(images).mean(1)), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.txt_head(self.get_word_tokens(tokens).mean(1)), dim=-1)


B = 4
images = torch.randn(B, 3, 224, 224)
tokens = torch.randint(1, 30000, (B, L))
tokens[:, 200:] = 0   # PAD tail

model = AFRNet(MockBackbone(), hidden_dim=D_HID, proj_dim=D_PROJ, pad_token_id=0)
loss_fn = AFRLoss(LossConfig(ot_weight=0.1, ot_warmup_steps=100, distill_weight=0.1))

# --- retrieval interface (decoupled) ---
gi = model.encode_image(images)
gt = model.encode_text(tokens)
assert gi.shape == (B, D_PROJ) and gt.shape == (B, D_PROJ)
assert torch.allclose(gi.norm(dim=-1), torch.ones(B), atol=1e-4)

# --- training forward ---
out = model(images, tokens)
assert out["v_img"].shape == (B, D_PROJ)
assert out["t_txt"].shape == (B, D_PROJ)
assert out["P_star"].shape[0] == B

L_total, parts = loss_fn(
    out["v_img"], out["t_txt"], out["L_OT"], step=50,
    v_cls_frozen=out["v_cls_frozen"], t_frozen=out["t_frozen"],
)
L_total.backward()

# every AFR-Net submodule must receive gradient (no dead modules)
def grad_alive(module):
    ps = [p for p in module.parameters() if p.requires_grad]
    return len(ps) > 0 and any(p.grad is not None and p.grad.abs().sum() > 0 for p in ps)

status = {name: grad_alive(getattr(model, name))
          for name in ["roam", "gtga", "ddtp", "img_proj", "txt_proj"]}
print("loss parts:", {k: (round(v.item(), 4) if hasattr(v, 'item') else v) for k, v in parts.items()})
print("module grad-alive:", status)
assert all(status.values()), f"DEAD MODULE: {status}"
print(f"L_total = {L_total.item():.4f}")
print("PASS — AFR-Net full forward/backward, all modules live")
