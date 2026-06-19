"""
TEST 3 — open_clip & BiomedCLIP Weights
Run this after: uv add open-clip-torch
This will download BiomedCLIP weights on first run (~1.5 GB, cached after).
Expected: model loads, image + text encoders produce normalized vectors
"""

print("=" * 50)
print("TEST 3: open_clip & BiomedCLIP")
print("=" * 50)

try:
    import open_clip
    print(f"open_clip version: {open_clip.__version__}")

    print("\nLoading BiomedCLIP weights (downloads ~1.5 GB on first run)...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    print("Weights loaded ✅")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test image encoding
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        img_features = model.encode_image(dummy_image)

    import torch.nn.functional as F
    img_features = F.normalize(img_features, dim=-1)
    print(f"\nImage encoder output shape : {img_features.shape}")
    print(f"Image feature norm (should be ~1.0): {img_features.norm(dim=-1).mean():.4f}")

    # Test text encoding
    texts = ["annular erythematous plaque with central clearing",
             "right lower lobe consolidation with air bronchograms"]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        txt_features = model.encode_text(tokens)
    txt_features = F.normalize(txt_features, dim=-1)
    print(f"Text encoder output shape  : {txt_features.shape}")
    print(f"Text feature norm (should be ~1.0): {txt_features.norm(dim=-1).mean():.4f}")

    # Test similarity
    sim = img_features @ txt_features.T
    print(f"\nSimilarity matrix:\n{sim}")

    print("\n✅ PASS — BiomedCLIP is fully working")

except ImportError as e:
    print(f"❌ FAIL — Missing package: {e}")
    print("   Fix: uv add open-clip-torch")
except Exception as e:
    print(f"❌ FAIL — Error: {e}")
