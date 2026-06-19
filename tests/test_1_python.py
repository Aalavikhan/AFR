"""
TEST 1 — Python Version
Run this first, before installing anything.
Expected: Python 3.10 or higher
"""

import sys

print("=" * 50)
print("TEST 1: Python Version")
print("=" * 50)

major, minor = sys.version_info.major, sys.version_info.minor
print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")

if major == 3 and minor >= 10:
    print("\n✅ PASS — Python 3.10+ detected")
else:
    print(f"\n❌ FAIL — Need Python 3.10+, got {major}.{minor}")
    print("   Fix: uv python install 3.10")
    print("        uv venv --python 3.10")
