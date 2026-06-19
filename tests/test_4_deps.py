"""
TEST 4 — All Other Dependencies
Run this after installing everything:
  uv add transformers==4.40.0 timm einops scipy scikit-learn albumentations POT pandas pillow tqdm pyyaml
Expected: all imports succeed, version table printed
"""

print("=" * 50)
print("TEST 4: All Dependencies")
print("=" * 50)

results = []

def check(label, fn):
    try:
        version = fn()
        results.append((label, "✅", version))
    except Exception as e:
        results.append((label, "❌", str(e)))

# --- Core ---
check("torch", lambda: __import__("torch").__version__)
check("torchvision", lambda: __import__("torchvision").__version__)
check("open_clip", lambda: __import__("open_clip").__version__)
check("transformers", lambda: __import__("transformers").__version__)
check("timm", lambda: __import__("timm").__version__)

# --- Math / Science ---
check("einops", lambda: __import__("einops").__version__)
check("scipy", lambda: __import__("scipy").__version__)
check("scikit-learn", lambda: __import__("sklearn").__version__)
check("POT (ot)", lambda: __import__("ot").__version__)
check("numpy", lambda: __import__("numpy").__version__)

# --- Image / Data ---
check("albumentations", lambda: __import__("albumentations").__version__)
check("PIL (Pillow)", lambda: __import__("PIL").__version__)
check("pandas", lambda: __import__("pandas").__version__)

# --- Utils ---
check("tqdm", lambda: __import__("tqdm").__version__)
check("yaml (pyyaml)", lambda: __import__("yaml").__version__)
check("pathlib", lambda: "built-in")

# --- Print table ---
print(f"\n{'Package':<20} {'Status':<6} {'Version / Error'}")
print("-" * 65)
for label, status, version in results:
    print(f"{label:<20} {status:<6} {version}")

failures = [r for r in results if r[1] == "❌"]
print()
if not failures:
    print("✅ ALL PASS — environment is complete")
else:
    print(f"❌ {len(failures)} package(s) failed:")
    for label, _, err in failures:
        print(f"   {label}: {err}")
    print("\nFix: uv add <package-name>")
    print("For transformers specifically: uv add transformers==4.40.0")
