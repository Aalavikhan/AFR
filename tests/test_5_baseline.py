"""
TEST 5 — Baseline Model End-to-End
Run this after test_3 passes. Tests the full baseline forward pass,
loss computation, and a single backprop step.
Expected: loss printed, gradients flow, no errors
"""

print("=" * 50)
print("TEST 5: Baseline Model End-to-End")
print("=" * 50)

import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load model ---
    print("Loading BiomedCLIP...")
    model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model = model.to(device)
    model.train()

    # --- Dummy batch ---
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    captions = [
        "annular erythematous plaque with central clearing",
        "right lower lobe consolidation with air bronchograms",
        "hyperpigmented patch on lower extremity",
        "bilateral interstitial infiltrates suggestive of pneumonia",
    ]
    tokens = tokenizer(captions).to(device)

    # --- Forward pass ---
    print(f"Running forward pass (batch_size={batch_size})...")
    img_features = F.normalize(model.encode_image(images), dim=-1)
    txt_features = F.normalize(model.encode_text(tokens), dim=-1)

    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * img_features @ txt_features.T
    logits_per_text = logits_per_image.T

    print(f"Logits shape: {logits_per_image.shape}")  # [4, 4]

    # --- CLIP loss ---
    labels = torch.arange(batch_size, device=device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i2t + loss_t2i) / 2

    print(f"Loss i2t : {loss_i2t.item():.4f}")
    print(f"Loss t2i : {loss_t2i.item():.4f}")
    print(f"Total loss: {loss.item():.4f}  (expect ~{torch.log(torch.tensor(float(batch_size))):.2f} at random init)")

    # --- Backward pass ---
    print("\nRunning backward pass...")
    loss.backward()

    # Check gradients exist
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    print(f"Parameters with gradients: {len(grad_norms)}")
    print(f"Mean gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")

    if len(grad_norms) > 0:
        print("\n✅ PASS — baseline model forward + backward working")
    else:
        print("\n❌ FAIL — no gradients computed")

    # --- Memory report ---
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU memory allocated : {allocated:.2f} GB")
        print(f"GPU memory reserved  : {reserved:.2f} GB")

except Exception as e:
    import traceback
    print(f"\n❌ FAIL")
    traceback.print_exc()
