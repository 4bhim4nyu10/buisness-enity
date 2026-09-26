"""Choosing the decision rule empirically.

There are three families of decision rule worth considering, and which one wins
is an empirical question, not a matter of principle:

  1. `threshold`      — one global probability cut.
  2. `threshold_pair` — a lenient cut for the first candidate, a strict cut for
                        every additional one. A crude approximation of the
                        metric's true break-even structure.
  3. `expected_f`     — exact expected-F_0.5 set selection (see src/decide.py).

Simulation says these land within a couple of points of each other when
probabilities are well calibrated, with the ordering flipping depending on the
singleton rate and classifier quality. So we search all three on out-of-fold
predictions and keep the winner, rather than arguing about it.

One caveat worth respecting: this search selects on the validation data, so the
winning number is optimistic. Confirm the choice on a country holdout before
trusting it — see `--holdout-country` in src/train.py.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, replace
from typing import Iterable

import numpy as np

from .decide import (
    DecisionConfig,
    apply_strategy,
    decide_all,
    decide_threshold,
    decide_threshold_pair,
    decide_top1,
    enforce_unique_assignment,
)
from .evaluate import macro_f_beta, score_breakdown


def _score(spec: dict, sbe, truth, enforce_unique: bool) -> float:
    preds = apply_strategy(sbe, spec)
    if enforce_unique:
        preds = enforce_unique_assignment(preds, sbe, spec)
    return macro_f_beta(preds, truth)


def select_strategy(
    scores_by_entity: dict[str, list[tuple[str, float]]],
    truth: dict[str, set[str]],
    enforce_unique: bool = True,
    verbose: bool = True,
) -> tuple[dict, float, list[tuple[str, float]]]:
    """Search all three families. Returns (best spec, best score, full trace)."""
    trace: list[tuple[str, float]] = []
    best_spec: dict = {"strategy": "expected_f", **asdict(DecisionConfig())}
    best = -1.0

    # --- family 1: single global threshold -------------------------------
    for thr in np.arange(0.20, 0.925, 0.025):
        spec = {"strategy": "threshold", "thr": round(float(thr), 4)}
        s = _score(spec, scores_by_entity, truth, enforce_unique)
        trace.append((f"threshold {thr:.3f}", s))
        if s > best:
            best, best_spec = s, spec

    # --- family 2: first/rest threshold pair ------------------------------
    for tf in np.arange(0.20, 0.80, 0.05):
        for tr in np.arange(max(0.35, tf), 0.95, 0.05):
            spec = {
                "strategy": "threshold_pair",
                "thr_first": round(float(tf), 4),
                "thr_rest": round(float(tr), 4),
            }
            s = _score(spec, scores_by_entity, truth, enforce_unique)
            trace.append((f"pair {tf:.2f}/{tr:.2f}", s))
            if s > best:
                best, best_spec = s, spec

    # --- family 3: exact expected-F set selection -------------------------
    base = DecisionConfig()
    for t, sh, d, mp in itertools.product(
        (0.8, 1.0, 1.25, 1.5),
        (-0.75, -0.4, 0.0, 0.4),
        (0.85, 1.0),
        (0.02, 0.10, 0.20),
    ):
        cfg = replace(
            base, temperature=t, prior_shift=sh, independence_damping=d, min_prob=mp
        )
        spec = {"strategy": "expected_f", **asdict(cfg)}
        s = _score(spec, scores_by_entity, truth, enforce_unique)
        trace.append((f"expF T={t} shift={sh} damp={d} floor={mp}", s))
        if s > best:
            best, best_spec = s, spec

    if verbose:
        trace_sorted = sorted(trace, key=lambda kv: -kv[1])
        print("  best of each family:")
        for fam in ("threshold ", "pair ", "expF "):
            top = next((k for k, _ in trace_sorted if k.startswith(fam)), None)
            if top:
                v = dict(trace)[top]
                print(f"    {top:<44} {v:.4f}")
        print(f"  winner: {best_spec.get('strategy')} -> {best:.4f}")
    return best_spec, best, trace


# Backwards-compatible helper for callers that only want the expected-F family.
def tune_decision(
    scores_by_entity,
    truth,
    base: DecisionConfig | None = None,
    temperatures: Iterable[float] = (0.8, 1.0, 1.25, 1.5),
    shifts: Iterable[float] = (-0.75, -0.4, 0.0, 0.4),
    dampings: Iterable[float] = (0.85, 1.0),
    floors: Iterable[float] = (0.02, 0.10, 0.20),
    enforce_unique: bool = True,
    verbose: bool = False,
) -> tuple[DecisionConfig, float]:
    base = base or DecisionConfig()
    best_cfg, best_score = base, -1.0
    for t, s, d, f in itertools.product(temperatures, shifts, dampings, floors):
        cfg = replace(base, temperature=t, prior_shift=s, independence_damping=d, min_prob=f)
        sc = _score({"strategy": "expected_f", **asdict(cfg)}, scores_by_entity, truth, enforce_unique)
        if verbose:
            print(f"  T={t:<5} shift={s:<5} damp={d:<5} floor={f:<5} -> {sc:.4f}")
        if sc > best_score:
            best_cfg, best_score = cfg, sc
    return best_cfg, best_score


# --------------------------------------------------------------------------- #


def ablation_table(
    scores_by_entity: dict[str, list[tuple[str, float]]],
    truth: dict[str, set[str]],
    spec: dict,
    thresholds: Iterable[float] = (0.3, 0.5, 0.7, 0.85),
) -> dict[str, float]:
    """The table for your methodology document. Report it honestly — including
    the case where a plain threshold wins."""
    rows: dict[str, float] = {}
    for thr in thresholds:
        rows[f"global threshold {thr:.2f}"] = macro_f_beta(
            decide_threshold(scores_by_entity, thr), truth
        )
        rows[f"top-1 only if >= {thr:.2f}"] = macro_f_beta(
            decide_top1(scores_by_entity, thr), truth
        )
    rows["predict nothing for everyone"] = macro_f_beta(
        {k: [] for k in scores_by_entity}, truth
    )
    rows["all candidates (blocking ceiling)"] = macro_f_beta(
        {k: [t for t, _ in v] for k, v in scores_by_entity.items()}, truth
    )
    preds = apply_strategy(scores_by_entity, spec)
    label = f"selected rule: {spec.get('strategy')}"
    rows[label] = macro_f_beta(preds, truth)
    rows[f"  {label} + unique-target"] = macro_f_beta(
        enforce_unique_assignment(preds, scores_by_entity, spec), truth
    )
    return rows


def print_ablation(rows: dict[str, float]) -> None:
    width = max(len(k) for k in rows)
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<{width}}  {v:.4f}")


def report(preds, truth) -> None:
    print(score_breakdown(preds, truth))
