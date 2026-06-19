"""
TEST 2 — CUDA & GPU
Run this after: uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Expected: CUDA available, GPU name shown, tensor ops working
"""

print("=" * 50)
print("TEST 2: CUDA & GPU")
print("=" * 50)

try:
    import torch
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"GPU count       : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            print(f"GPU {i}          : {props.name} ({vram_gb:.1f} GB VRAM)")

        # Functional test — create tensors on GPU and multiply
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = a @ b
        print(f"\nMatrix multiply (1000x1000) on GPU: {c.shape} ✅")

        # Check if A100 / enough VRAM for training
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 35:
            print(f"✅ PASS — GPU has {vram_gb:.1f} GB VRAM (sufficient for AFR-Net)")
        elif vram_gb >= 16:
            print(f"⚠️  WARN — GPU has {vram_gb:.1f} GB VRAM (reduce batch_size to 16-32)")
        else:
            print(f"❌ WARN — GPU has only {vram_gb:.1f} GB VRAM (reduce batch_size to 8, may be slow)")
    else:
        print("\n❌ FAIL — CUDA not available")
        print("   Possible fixes:")
        print("   1. Check nvidia-smi works in PowerShell")
        print("   2. Make sure you installed the cu121 build of PyTorch:")
        print("      uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   3. Your GPU driver may be too old — update from nvidia.com")

except ImportError:
    print("❌ FAIL — PyTorch not installed")
    print("   Fix: uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
