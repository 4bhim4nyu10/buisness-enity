"""The official metric, plus the validation protocol.

Two things matter here:

1. Implement macro F_0.5 *exactly* as the organisers describe it — including
   singletons scoring 1.0 for a correct empty prediction. Everything you tune
   is tuned against this number, so a subtly wrong scorer is worse than none.

2. Validate with a **country holdout**. The test set contains France, which
   never appears in training. A random split will lie to you about how well the
   pipeline generalises; training on US and validating on India (and the other
   way round) is the only honest proxy you have for the France gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

BETA2 = 0.25


def entity_f_beta(pred: Iterable[str], gold: Iterable[str]) -> float:
    pred_s, gold_s = set(pred), set(gold)
    if not gold_s:
        return 1.0 if not pred_s else 0.0
    if not pred_s:
        return 0.0
    tp = len(pred_s & gold_s)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_s)
    recall = tp / len(gold_s)
    return (1 + BETA2) * precision * recall / (BETA2 * precision + recall)


def macro_f_beta(
    predictions: dict[str, Iterable[str]],
    truth: dict[str, Iterable[str]],
) -> float:
    """Averaged over **all** Source-1 entities in `truth` — an entity missing
    from `predictions` counts as an empty prediction, exactly as the grader
    would treat a missing row (it would actually reject the file; this is the
    charitable version so you notice the gap in the score)."""
    if not truth:
        return 0.0
    return float(
        np.mean([entity_f_beta(predictions.get(sid, []), gold) for sid, gold in truth.items()])
    )


@dataclass
class ScoreBreakdown:
    overall: float
    singletons: float
    single_match: float
    multi_match: float
    n_singletons: int
    n_single: int
    n_multi: int
    mean_pred_size: float
    mean_true_size: float

    def __str__(self) -> str:
        return (
            f"macro F0.5 = {self.overall:.4f}\n"
            f"  singletons   ({self.n_singletons:>6}) : {self.singletons:.4f}\n"
            f"  1 true match ({self.n_single:>6}) : {self.single_match:.4f}\n"
            f"  2+ matches   ({self.n_multi:>6}) : {self.multi_match:.4f}\n"
            f"  mean predicted set = {self.mean_pred_size:.2f}, "
            f"mean true set = {self.mean_true_size:.2f}"
        )


def score_breakdown(
    predictions: dict[str, Iterable[str]],
    truth: dict[str, Iterable[str]],
) -> ScoreBreakdown:
    """Always look at this, never just the headline number.

    The three strata fail in different ways and want different fixes:
    a low singleton score means the decision layer is too greedy; a low
    multi-match score usually means blocking recall, not the model.
    """
    buckets: dict[str, list[float]] = {"zero": [], "one": [], "many": []}
    psz, tsz = [], []
    for sid, gold in truth.items():
        gold_s = set(gold)
        pred = set(predictions.get(sid, []))
        f = entity_f_beta(pred, gold_s)
        psz.append(len(pred))
        tsz.append(len(gold_s))
        key = "zero" if not gold_s else ("one" if len(gold_s) == 1 else "many")
        buckets[key].append(f)

    def avg(x: Sequence[float]) -> float:
        return float(np.mean(x)) if len(x) else 0.0

    allv = buckets["zero"] + buckets["one"] + buckets["many"]
    return ScoreBreakdown(
        overall=avg(allv),
        singletons=avg(buckets["zero"]),
        single_match=avg(buckets["one"]),
        multi_match=avg(buckets["many"]),
        n_singletons=len(buckets["zero"]),
        n_single=len(buckets["one"]),
        n_multi=len(buckets["many"]),
        mean_pred_size=avg(psz),
        mean_true_size=avg(tsz),
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def country_holdout(
    s1_country: dict[str, str], holdout: str
) -> tuple[list[str], list[str]]:
    """Train on every country except `holdout`; validate on `holdout`.

    This is your France simulator. If the score on a held-out country collapses
    relative to a random split, your features are memorising country-specific
    surface patterns and will not transfer.
    """
    train = [sid for sid, c in s1_country.items() if c != holdout]
    val = [sid for sid, c in s1_country.items() if c == holdout]
    return train, val


def grouped_kfold(entity_ids: Sequence[str], n_folds: int = 5, seed: int = 13) -> list[np.ndarray]:
    """Fold assignment by Source-1 entity. Splitting by *pair* would leak: the
    relational features of a pair depend on its sibling candidates."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(entity_ids))
    rng.shuffle(idx)
    return [idx[i::n_folds] for i in range(n_folds)]
