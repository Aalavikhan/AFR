"""
TEST 6 — Retrieval Evaluation (R@K, MRR)
Runs the evaluator on a tiny synthetic dataset.
Expected: R@1/5/10 and MRR printed, values near 1.0 (since
          synthetic data is matched perfectly by construction)
"""

print("=" * 50)
print("TEST 6: Retrieval Evaluation")
print("=" * 50)

import torch
import torch.nn.functional as F


def evaluate_retrieval(img_feats, txt_feats, ks=(1, 5, 10)):
    """
    img_feats: [N, D] normalized
    txt_feats: [N, D] normalized
    Returns dict of i2t and t2i R@k and MRR
    """
    N = img_feats.shape[0]
    sim = img_feats @ txt_feats.T  # [N, N]
    results = {}

    for k in ks:
        k_eff = min(k, N)

        # i2t
        topk_i2t = sim.topk(k_eff, dim=1).indices
        i2t_hits = sum(i in row.tolist() for i, row in enumerate(topk_i2t))
        results[f'i2t_R@{k}'] = i2t_hits / N

        # t2i
        topk_t2i = sim.T.topk(k_eff, dim=1).indices
        t2i_hits = sum(i in row.tolist() for i, row in enumerate(topk_t2i))
        results[f't2i_R@{k}'] = t2i_hits / N

    # MRR
    ranks_i2t = (sim.argsort(dim=1, descending=True) == torch.arange(N).unsqueeze(1)).float().argmax(dim=1) + 1
    ranks_t2i = (sim.T.argsort(dim=1, descending=True) == torch.arange(N).unsqueeze(1)).float().argmax(dim=1) + 1
    results['i2t_MRR'] = (1.0 / ranks_i2t.float()).mean().item()
    results['t2i_MRR'] = (1.0 / ranks_t2i.float()).mean().item()

    return results


# --- Perfect alignment test ---
# Each image feature is identical to its paired text feature
# R@1 should be 1.0 (perfect)
N, D = 50, 512
base = torch.randn(N, D)
img_feats = F.normalize(base + 0.01 * torch.randn(N, D), dim=-1)
txt_feats = F.normalize(base + 0.01 * torch.randn(N, D), dim=-1)

print(f"N={N} samples, D={D} dimensions")
print(f"Synthetic: paired features nearly identical (expect high scores)\n")

results = evaluate_retrieval(img_feats, txt_feats)

print(f"{'Metric':<15} {'Score':>8}")
print("-" * 25)
for k, v in results.items():
    print(f"{k:<15} {v:>8.4f}")

# Sanity check
if results['i2t_R@1'] > 0.9:
    print("\n✅ PASS — evaluator working correctly")
else:
    print("\n❌ FAIL — R@1 too low for perfectly matched features; check eval logic")

# --- Random baseline ---
print("\n--- Random baseline (features completely unrelated) ---")
img_rand = F.normalize(torch.randn(N, D), dim=-1)
txt_rand = F.normalize(torch.randn(N, D), dim=-1)
rand_results = evaluate_retrieval(img_rand, txt_rand)
for k, v in rand_results.items():
    print(f"{k:<15} {v:>8.4f}")
print(f"(R@1 should be ~{1/N:.3f} for random)")
