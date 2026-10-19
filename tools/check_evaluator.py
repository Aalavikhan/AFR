"""
tools/check_evaluator.py
------------------------
Verifies the PRODUCTION retrieval evaluator (evaluators/retrieval_eval.py)
against hand-computed values.

Unlike tests/test_6_eval.py -- which defines its own private copy of the metric
logic and therefore proves nothing about production code -- this imports
RetrievalEvaluator itself and asserts exact numbers.

Needs only torch. No BiomedCLIP, no network, no dataset, CPU only:

    python -X utf8 tools/check_evaluator.py

Exit code 0 = all numeric checks passed. Exit code 1 = at least one failed.

Last executed 2026-10-18 on torch 2.13.0+cpu: 11/11 checks passed.
Documented behaviours this pins down (see docs/KNOWN_ISSUES.md):
  - R@K / MRR / median-rank arithmetic is correct (AFR-006 is about memory,
    not about the formulas).
  - torch.median returns the LOWER middle value for even N, not the average.
  - With N < k, R@K is reported under the requested name but holds R@N.
  - Strict diagonal ground truth understates recall when positives are not
    unique (AFR-011).
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evaluators.retrieval_eval import RetrievalEvaluator  # noqa: E402

DEVICE = torch.device("cpu")
_failures: list[str] = []


class StubModel:
    """Minimal duck-type. The evaluator needs only eval/encode_image/encode_text.

    The fake loader already yields feature tensors, so encoding is the identity.
    This keeps the check free of BiomedCLIP and of any network access.
    """

    def eval(self) -> None:
        pass

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return x


def fake_loader(img: torch.Tensor, txt: torch.Tensor, batch_size: int):
    """Mimic the (images, tokens) 2-tuple contract every consumer relies on."""
    return [
        (img[i:i + batch_size], txt[i:i + batch_size])
        for i in range(0, len(img), batch_size)
    ]


def metrics_for(img: torch.Tensor, txt: torch.Tensor) -> dict[str, float]:
    return RetrievalEvaluator(StubModel(), DEVICE).compute_metrics(img, txt)


def check(label: str, got: float, want: float, tol: float = 1e-6) -> None:
    passed = abs(got - want) <= tol
    if not passed:
        _failures.append(label)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: got {got!r}, want {want!r}")


def main() -> int:
    print("1) Perfect alignment (N=8, orthonormal rows)")
    feats = F.normalize(torch.eye(8), dim=-1)
    m = metrics_for(feats, feats)
    check("i2t_R@1", m["i2t_R@1"], 1.0)
    check("t2i_R@1", m["t2i_R@1"], 1.0)
    check("i2t_MRR", m["i2t_MRR"], 1.0)
    check("i2t_median_rank", m["i2t_median_rank"], 1.0)

    print("\n2) Captions 0 and 1 exchanged -> ranks are exactly [2, 2, 1, 1]")
    img = F.normalize(torch.eye(4), dim=-1)
    txt = img.clone()[[1, 0, 2, 3]]
    m = metrics_for(img, txt)
    check("i2t_R@1 (2 of 4 wrong)", m["i2t_R@1"], 0.5)
    check("i2t_R@5 (all found)", m["i2t_R@5"], 1.0)
    check("i2t_MRR = (1/2 + 1/2 + 1 + 1) / 4", m["i2t_MRR"], 0.75)
    # torch.median takes the lower of the two middle values for even N,
    # so median([1, 1, 2, 2]) is 1.0 and not the statistical 1.5.
    check("i2t_median_rank (lower middle, not 1.5)", m["i2t_median_rank"], 1.0)
    ranks = torch.tensor([2.0, 2.0, 1.0, 1.0])
    print(f"     torch.median={ranks.median().item()}  "
          f"quantile(0.5)={torch.quantile(ranks, 0.5).item()}")

    print("\n3) N < k -> R@K keeps the requested name but holds R@N")
    feats = F.normalize(torch.eye(3), dim=-1)
    m = metrics_for(feats, feats)
    print(f"     i2t_R@5={m['i2t_R@5']}  i2t_R@10={m['i2t_R@10']}  (both are R@3)")
    check("i2t_R@10 present on N=3", m["i2t_R@10"], 1.0)

    print("\n4) evaluate() end to end through the loader contract")
    feats = F.normalize(torch.randn(20, 16), dim=-1)
    m = RetrievalEvaluator(StubModel(), DEVICE).evaluate(fake_loader(feats, feats, 6))
    check("evaluate() i2t_R@1", m["i2t_R@1"], 1.0)
    check("metric count", float(len(m)), 10.0)

    print("\n5) Empty loader")
    try:
        RetrievalEvaluator(StubModel(), DEVICE).evaluate([])
        print("  [INFO] empty loader did NOT raise")
    except Exception as exc:  # noqa: BLE001 - reporting the observed failure mode
        print(f"  [INFO] raises {type(exc).__name__}: {str(exc)[:80]}")

    print("\n6) Two images sharing an identical caption (true multi-positive)")
    # Images 0 and 1 are identical and so are captions 0 and 1. Retrieving
    # caption 0 for image 1 is correct in reality, but strict diagonal scoring
    # demands column 1, and the tie breaks to column 0 -> counted as an error.
    eye = torch.eye(4)
    img = F.normalize(torch.stack([eye[0], eye[0], eye[2], eye[3]]), dim=-1)
    m = metrics_for(img, img.clone())
    print(f"     strict diagonal i2t_R@1 = {m['i2t_R@1']:.4f}, "
          f"grouped-positive truth = 1.0000")
    check("strict diagonal understates R@1", m["i2t_R@1"], 0.75)

    print("\n7) O(N^2) growth of the argsort rank path (AFR-006)")
    for n in (500, 1500):
        feats = F.normalize(torch.randn(n, 32), dim=-1)
        metrics_for(feats, feats)
        sim_mb = n * n * 4 / 1e6
        print(f"     N={n:5d} ok: sim={sim_mb:7.1f} MB, "
              f"each int64 argsort={sim_mb * 2:7.1f} MB")
    print("     -> at N=9900: sim=392 MB, each argsort=784 MB, two live at once")

    if _failures:
        print(f"\n{len(_failures)} CHECK(S) FAILED: {', '.join(_failures)}")
        return 1
    print("\nALL NUMERIC CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
