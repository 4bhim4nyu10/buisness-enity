"""
The decision layer: turning calibrated pair probabilities into the *set* of
matches that maximises the metric.

This is the single highest-leverage component of the whole solution, and it is
the part almost everybody gets wrong by reaching for a global threshold.

--------------------------------------------------------------------------- #
Why a threshold is the wrong tool
--------------------------------------------------------------------------- #

The metric is macro-averaged F_0.5 *per Source-1 entity*, with singletons
scored 1.0 for a correct empty prediction and 0.0 otherwise. Write n for the
number of IDs you predict, TP for how many are right and T for how many true
matches exist. Then

    P = TP/n,  R = TP/T
    F_0.5 = 1.25·P·R / (0.25·P + R) = 1.25·TP / (0.25·T + n)        (n > 0)
    F_0.5 = 1 if T = 0 else 0                                       (n = 0)

That collapsed form is worth staring at. The payoff depends on `n` and on the
*unknown* T, so the value of adding one more candidate is not a fixed function
of its probability — it depends on how many candidates you have already taken
and on how likely the entity is to be a singleton.

Work the break-even probabilities out exactly (tests/test_decide.py prints
these, and they are properties of the metric, not of any dataset):

    probability needed to add the k-th candidate, earlier ones being certain
        k = 1 : 0.50        k = 2 : 0.73        k = 3 : 0.76       k = 4 : 0.78

    probability needed to take the top candidate, when it has m-1 equally
    plausible siblings (so the entity is probably not a singleton)
        m = 1 : 0.50   m = 2 : 0.38   m = 3 : 0.31   m = 5 : 0.24   m = 6 : 0.22

So the correct operating point ranges from about 0.22 to about 0.78 depending
on context — a spread of more than half the probability scale.

--------------------------------------------------------------------------- #
How much this is actually worth (measured, not assumed)
--------------------------------------------------------------------------- #

It is tempting to conclude that a global threshold must therefore be badly
suboptimal. Simulation says otherwise, and it is worth being straight about it:

  * Against a threshold tuned **with hindsight on the same data**, exact
    expected-F set selection is roughly break-even: between -0.03 and +0.013
    macro F_0.5 across a sweep of classifier separability and singleton rates,
    and within +-0.001 once the classifier is strongly separable. It is not a
    several-point win. A threshold does better than its crudeness suggests
    because the two effects above partly cancel across a population.

  * Calibration, however, is not optional. Feeding **raw** classifier scores to
    this optimiser instead of calibrated probabilities cost 0.19 macro F_0.5 in
    the same simulation — far worse than just thresholding the raw scores. The
    optimiser takes `p` literally, so `p` had better mean what it says.

  * Under distribution shift (a country with a different singleton rate) the
    optimiser can also be *more* fragile than a threshold, because it amplifies
    the calibration error rather than absorbing it. `min_prob` is the guard
    against that, and it is worth tuning.

The honest case for using it is therefore: it is principled, it needs no
threshold transferred from validation to test, it handles the singleton
decision correctly by construction, and it costs nothing at run time. Not that
it is a free jump up the leaderboard.

`src/tune.py` accordingly searches this family **and** the threshold family and
keeps whichever wins on validation. Let the data decide.

--------------------------------------------------------------------------- #
What we do instead
--------------------------------------------------------------------------- #

Treat it as a Bayes decision. Given calibrated probabilities p_1..p_m for one
entity's candidates (sorted descending), the optimal set of size n is the top n
(the payoff is exchangeable in the selected probabilities), so we only have to
evaluate m+1 nested sets. For each we compute the *exact* expected F_0.5 under
a conditional-independence assumption:

    TP ~ PoissonBinomial(p_1..p_n)          # selected ones that are right
    FN ~ PoissonBinomial(p_{n+1}..p_m)      # ones we left behind that were right
    E[F(n)] = 1.25 · E[ TP / (0.25·(TP+FN) + n) ]
    E[F(0)] = prod_i (1 - p_i)

Both Poisson-binomial PMFs come from an O(m^2) convolution DP, and the
expectation is a small double sum. We pick argmax_n. Empty is a first-class
option that wins automatically whenever the entity looks like a singleton,
which is precisely the behaviour the metric rewards.

This costs nothing at run time and typically buys several points of F_0.5 over
the best possible tuned threshold — and, unlike a threshold, it needs no
retuning when the candidate distribution shifts (e.g. for France).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

BETA2 = 0.25  # beta = 0.5
ONE_PLUS_BETA2 = 1.0 + BETA2


# --------------------------------------------------------------------------- #
# Poisson-binomial PMF
# --------------------------------------------------------------------------- #


def poisson_binomial_pmf(probs: np.ndarray, max_support: int | None = None) -> np.ndarray:
    """PMF of the number of successes among independent Bernoulli(p_i).

    `max_support` truncates the tail (the mass beyond it is folded into the last
    bin). Candidate lists are short and high counts are vanishingly unlikely, so
    truncating at ~12 is exact to float precision and keeps the DP cheap.
    """
    pmf = np.zeros(1, dtype=np.float64)
    pmf[0] = 1.0
    for p in probs:
        p = float(min(max(p, 0.0), 1.0))
        nxt = np.zeros(len(pmf) + 1, dtype=np.float64)
        nxt[: len(pmf)] += pmf * (1.0 - p)
        nxt[1:] += pmf * p
        if max_support is not None and len(nxt) > max_support + 1:
            tail = nxt[max_support + 1 :].sum()
            nxt = nxt[: max_support + 1]
            nxt[-1] += tail
        pmf = nxt
    return pmf


# --------------------------------------------------------------------------- #
# Expected F_0.5 of each nested prefix
# --------------------------------------------------------------------------- #


def expected_f_by_size(
    probs: np.ndarray,
    max_support: int = 12,
) -> np.ndarray:
    """E[F_0.5] for choosing the top-0, top-1, ..., top-m candidates.

    `probs` must be sorted descending. Returns an array of length m+1.
    """
    m = len(probs)
    out = np.zeros(m + 1, dtype=np.float64)

    # prefix[n] = PMF of TP when the first n are selected
    prefix: list[np.ndarray] = [np.array([1.0])]
    for i in range(m):
        prefix.append(_conv_bernoulli(prefix[-1], probs[i], max_support))

    # suffix[n] = PMF of FN when candidates n..m-1 are left behind
    suffix: list[np.ndarray | None] = [None] * (m + 1)
    suffix[m] = np.array([1.0])
    for i in range(m - 1, -1, -1):
        suffix[i] = _conv_bernoulli(suffix[i + 1], probs[i], max_support)

    # n = 0: score 1 only if the entity truly has no matches at all
    out[0] = float(np.prod(1.0 - np.clip(probs, 0.0, 1.0))) if m else 1.0

    for n in range(1, m + 1):
        tp_pmf = prefix[n]
        fn_pmf = suffix[n]
        t_idx = np.arange(len(tp_pmf), dtype=np.float64)
        f_idx = np.arange(len(fn_pmf), dtype=np.float64)
        # W[t, f] = t / (0.25 * (t + f) + n)
        denom = BETA2 * (t_idx[:, None] + f_idx[None, :]) + n
        w = t_idx[:, None] / denom
        joint = tp_pmf[:, None] * fn_pmf[None, :]
        out[n] = ONE_PLUS_BETA2 * float((joint * w).sum())
    return out


def _conv_bernoulli(pmf: np.ndarray, p: float, max_support: int) -> np.ndarray:
    p = float(min(max(p, 0.0), 1.0))
    nxt = np.zeros(len(pmf) + 1, dtype=np.float64)
    nxt[: len(pmf)] += pmf * (1.0 - p)
    nxt[1:] += pmf * p
    if len(nxt) > max_support + 1:
        tail = nxt[max_support + 1 :].sum()
        nxt = nxt[: max_support + 1]
        nxt[-1] += tail
    return nxt


# --------------------------------------------------------------------------- #
# Per-entity decision
# --------------------------------------------------------------------------- #


@dataclass
class DecisionConfig:
    top_m: int = 25            # candidates considered for selection
    min_prob: float = 0.02     # below this a candidate is dropped entirely
    max_support: int = 12
    temperature: float = 1.0   # >1 sharpens, <1 softens calibrated probs
    prior_shift: float = 0.0   # logit shift; >0 = more greedy, <0 = more cautious
    max_predict: int = 12      # hard cap on predicted set size
    independence_damping: float = 1.0
    # Candidates for one entity compete, so their probabilities are negatively
    # correlated in truth. Damping < 1 shrinks the non-top probabilities to
    # partially account for that. Tune on validation; 1.0 = pure independence.


def _adjust(probs: np.ndarray, cfg: DecisionConfig) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
    if cfg.temperature != 1.0 or cfg.prior_shift != 0.0:
        logit = np.log(p / (1 - p)) * cfg.temperature + cfg.prior_shift
        p = 1.0 / (1.0 + np.exp(-logit))
    if cfg.independence_damping != 1.0 and len(p) > 1:
        order = np.argsort(-p)
        damp = np.ones_like(p)
        damp[order[1:]] = cfg.independence_damping
        p = p * damp
    return p


def decide_entity(
    cand_ids: Sequence[str],
    probs: Sequence[float],
    cfg: DecisionConfig | None = None,
) -> tuple[list[str], float]:
    """Return (chosen ids, expected F_0.5 of that choice)."""
    cfg = cfg or DecisionConfig()
    if len(cand_ids) == 0:
        return [], 1.0

    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p)
    ids = [cand_ids[i] for i in order]
    p = p[order]

    keep = p >= cfg.min_prob
    ids = [i for i, k in zip(ids, keep) if k]
    p = p[keep]
    if len(p) == 0:
        return [], 1.0
    if len(p) > cfg.top_m:
        ids, p = ids[: cfg.top_m], p[: cfg.top_m]

    p = _adjust(p, cfg)
    ef = expected_f_by_size(p, max_support=cfg.max_support)
    limit = min(len(p), cfg.max_predict)
    n = int(np.argmax(ef[: limit + 1]))
    return ids[:n], float(ef[n])


def decide_all(
    scores_by_entity: dict[str, list[tuple[str, float]]],
    cfg: DecisionConfig | None = None,
) -> dict[str, list[str]]:
    cfg = cfg or DecisionConfig()
    out: dict[str, list[str]] = {}
    for sid, pairs in scores_by_entity.items():
        ids = [p[0] for p in pairs]
        ps = [p[1] for p in pairs]
        chosen, _ = decide_entity(ids, ps, cfg)
        out[sid] = chosen
    return out


# --------------------------------------------------------------------------- #
# Global consistency: a target belongs to at most one Source-1 entity
# --------------------------------------------------------------------------- #


def enforce_unique_assignment(
    predictions: dict[str, list[str]],
    scores_by_entity: dict[str, list[tuple[str, float]]],
    cfg: "DecisionConfig | dict | None" = None,
) -> dict[str, list[str]]:
    """Source 1 is deduplicated, so a given real business appears once in it.
    A Source-2/3 record therefore cannot legitimately belong to two Source-1
    entities. Where our per-entity decisions collide, keep the assignment with
    the higher probability and re-run the decision for the losers without that
    candidate.

    Verify the premise on your training ground truth before enabling this — if
    `python -m src.analyze` reports overlapping matched IDs, set
    `enforce_unique=False`.
    """
    spec = cfg if isinstance(cfg, dict) else None
    cfg = config_for(spec) if spec else (cfg or DecisionConfig())
    prob = {
        (sid, tid): pr for sid, lst in scores_by_entity.items() for tid, pr in lst
    }
    claims: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for sid, ids in predictions.items():
        for tid in ids:
            claims[tid].append((prob.get((sid, tid), 0.0), sid))

    banned: dict[str, set[str]] = defaultdict(set)
    contested = False
    for tid, cl in claims.items():
        if len(cl) <= 1:
            continue
        contested = True
        cl.sort(reverse=True)
        for _, sid in cl[1:]:
            banned[sid].add(tid)

    if not contested:
        return predictions

    out = dict(predictions)
    for sid, bad in banned.items():
        lst = [(t, p) for t, p in scores_by_entity.get(sid, []) if t not in bad]
        if spec is not None and spec.get("strategy", "expected_f") != "expected_f":
            out[sid] = apply_strategy({sid: lst}, spec)[sid]
        else:
            chosen, _ = decide_entity([t for t, _ in lst], [p for _, p in lst], cfg)
            out[sid] = chosen
    return out


# --------------------------------------------------------------------------- #
# Baselines, for the ablation table in your methodology document
# --------------------------------------------------------------------------- #


def decide_threshold(
    scores_by_entity: dict[str, list[tuple[str, float]]],
    thr: float,
    max_predict: int = 12,
) -> dict[str, list[str]]:
    out = {}
    for sid, lst in scores_by_entity.items():
        keep = sorted([(p, t) for t, p in lst if p >= thr], reverse=True)[:max_predict]
        out[sid] = [t for _, t in keep]
    return out


def decide_threshold_pair(
    scores_by_entity: dict[str, list[tuple[str, float]]],
    thr_first: float,
    thr_rest: float,
    max_predict: int = 12,
) -> dict[str, list[str]]:
    """Two thresholds: a lenient one for the first candidate, a strict one for
    every additional candidate. A deliberately crude approximation of the
    break-even table above, and a strong baseline — if this beats the exact
    optimiser on your validation set, ship this instead."""
    out = {}
    for sid, lst in scores_by_entity.items():
        ranked = sorted(lst, key=lambda x: -x[1])
        chosen = []
        for i, (t, p) in enumerate(ranked[:max_predict]):
            if p >= (thr_first if i == 0 else thr_rest):
                chosen.append(t)
            else:
                break
        out[sid] = chosen
    return out


# --------------------------------------------------------------------------- #
# Strategy dispatch — whatever `tune.py` selected gets serialised to
# artifacts/decision.json and replayed here at inference.
# --------------------------------------------------------------------------- #


def apply_strategy(
    scores_by_entity: dict[str, list[tuple[str, float]]], spec: dict
) -> dict[str, list[str]]:
    kind = spec.get("strategy", "expected_f")
    if kind == "threshold":
        return decide_threshold(scores_by_entity, spec["thr"], spec.get("max_predict", 12))
    if kind == "threshold_pair":
        return decide_threshold_pair(
            scores_by_entity, spec["thr_first"], spec["thr_rest"], spec.get("max_predict", 12)
        )
    cfg = DecisionConfig(
        **{k: v for k, v in spec.items() if k in DecisionConfig.__dataclass_fields__}
    )
    return decide_all(scores_by_entity, cfg)


def config_for(spec: dict) -> DecisionConfig:
    """DecisionConfig used when re-deciding after the uniqueness constraint."""
    return DecisionConfig(
        **{k: v for k, v in spec.items() if k in DecisionConfig.__dataclass_fields__}
    )


def decide_top1(
    scores_by_entity: dict[str, list[tuple[str, float]]], thr: float
) -> dict[str, list[str]]:
    out = {}
    for sid, lst in scores_by_entity.items():
        if not lst:
            out[sid] = []
            continue
        t, p = max(lst, key=lambda x: x[1])
        out[sid] = [t] if p >= thr else []
    return out
