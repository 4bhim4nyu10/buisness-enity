"""Correctness checks for the decision layer.

The expected-F_0.5 DP is the component with the most room for a silent, costly
bug, so it is checked against brute-force enumeration over every possible truth
assignment. Run with:  python tests/test_decide.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.decide import decide_entity, expected_f_by_size  # noqa: E402
from src.evaluate import entity_f_beta, macro_f_beta  # noqa: E402


def brute_force_expected_f(probs, n_select):
    """Enumerate all 2^m truth worlds and average the realised F_0.5."""
    m = len(probs)
    ids = [f"c{i}" for i in range(m)]
    chosen = ids[:n_select]
    total = 0.0
    for bits in itertools.product([0, 1], repeat=m):
        w = 1.0
        gold = []
        for b, p, i in zip(bits, probs, ids):
            w *= p if b else (1 - p)
            if b:
                gold.append(i)
        total += w * entity_f_beta(chosen, gold)
    return total


def test_dp_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(30):
        m = int(rng.integers(1, 8))
        probs = np.sort(rng.random(m))[::-1]
        ef = expected_f_by_size(np.asarray(probs), max_support=m)
        for n in range(m + 1):
            bf = brute_force_expected_f(probs, n)
            assert abs(ef[n] - bf) < 1e-9, (
                f"trial {trial} n={n}: DP={ef[n]:.12f} brute={bf:.12f} probs={probs}"
            )
    print("  DP == brute force on 30 random problems  OK")


def test_metric_formula():
    # the worked example from the problem statement
    f = entity_f_beta(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert abs(f - 0.714285714) < 1e-6, f
    # singletons
    assert entity_f_beta([], []) == 1.0
    assert entity_f_beta(["x"], []) == 0.0
    assert entity_f_beta([], ["x"]) == 0.0
    print("  metric matches the statement's worked example  OK")


def test_decision_behaviour():
    # a confident single candidate is taken
    ids, _ = decide_entity(["a"], [0.9])
    assert ids == ["a"], ids
    # a hopeless one is dropped -> predicted singleton
    ids, _ = decide_entity(["a"], [0.05])
    assert ids == [], ids
    # one strong + one weak: the weak one must NOT be added
    ids, _ = decide_entity(["a", "b"], [0.95, 0.25])
    assert ids == ["a"], ids
    # two strong: both taken
    ids, _ = decide_entity(["a", "b"], [0.9, 0.85])
    assert ids == ["a", "b"], ids
    print("  decision behaviour is precision-appropriate  OK")


def _breakeven_kth(k: int) -> float:
    """Probability the k-th candidate needs, given k-1 certain ones."""
    for p in np.arange(0.005, 1.0, 0.005):
        probs = np.array([1 - 1e-9] * (k - 1) + [float(p)])
        if int(np.argmax(expected_f_by_size(probs))) == k:
            return float(p)
    return 1.0


def _breakeven_top(m: int) -> float:
    """Probability the top candidate needs when it has m-1 equal siblings."""
    for p in np.arange(0.005, 1.0, 0.005):
        if int(np.argmax(expected_f_by_size(np.array([float(p)] * m)))) >= 1:
            return float(p)
    return 1.0


def test_operating_point_is_context_dependent():
    """The whole case for the decision layer: the optimal operating point moves
    over half the probability scale depending on context, so no single global
    threshold can be right."""
    k = [_breakeven_kth(i) for i in (1, 2, 3, 4)]
    m = [_breakeven_top(i) for i in (1, 2, 3, 5)]
    print("  break-even to add the k-th candidate: "
          + ", ".join(f"k={i}:{v:.2f}" for i, v in zip((1, 2, 3, 4), k)))
    print("  break-even for the top of m siblings: "
          + ", ".join(f"m={i}:{v:.2f}" for i, v in zip((1, 2, 3, 5), m)))
    assert 0.49 < k[0] < 0.52, k          # a lone candidate: exactly a coin flip
    assert k[1] > 0.70, k                 # the second needs to be far stronger
    assert k[1] < k[2] < k[3], k          # and it gets harder from there
    assert m[3] < 0.30, m                 # siblings make the top one cheaper
    assert max(k) - min(m) > 0.45, (k, m) # spread > half the scale
    print("  operating point spans "
          f"{min(m):.2f}..{max(k):.2f} — a global threshold cannot cover it  OK")


def test_macro_average():
    truth = {"e1": {"a"}, "e2": set(), "e3": {"a", "b"}}
    preds = {"e1": ["a"], "e2": [], "e3": ["a"]}
    expected = (1.0 + 1.0 + entity_f_beta(["a"], ["a", "b"])) / 3
    assert abs(macro_f_beta(preds, truth) - expected) < 1e-12
    print("  macro averaging includes singletons  OK")


if __name__ == "__main__":
    test_metric_formula()
    test_dp_matches_brute_force()
    test_decision_behaviour()
    test_operating_point_is_context_dependent()
    test_macro_average()
    print("\nall decision-layer tests passed")
