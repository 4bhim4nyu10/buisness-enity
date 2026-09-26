"""End-to-end orchestration: data -> normalisation -> blocking -> features ->
model -> decision -> submission files."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .blocking import BlockingResult, CountryBlocker, DEFAULT_CHANNELS, blocking_report
from .dataio import SourceSet, load_ground_truth, load_sources, write_candidate_pairs, write_matching_results
from .decide import DecisionConfig, apply_strategy, enforce_unique_assignment
from .evaluate import score_breakdown
from .features import Featurizer, Pair
from .model import TwoStageMatcher, make_labels
from .normalize import CorpusStats, NormRecord, build_record


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class PreparedSplit:
    s1: dict[str, NormRecord]
    targets: dict[str, NormRecord]
    source_of: dict[str, int]
    blocking: BlockingResult = field(default_factory=BlockingResult)
    n_targets: int = 0


# --------------------------------------------------------------------------- #


def fit_corpus_stats(*sets: SourceSet) -> CorpusStats:
    """Fit document frequencies over every record we are allowed to see —
    training *and* test. This is transductive use of the provided data, not an
    external lookup, and it is what makes an unseen country's legal suffixes and
    street words get demoted automatically."""
    names, addrs, countries = [], [], []
    for s in sets:
        if s is None:
            continue
        df = s.all_records()
        names.extend(df["business_name"].tolist())
        addrs.extend(df["business_address"].tolist())
        countries.extend(df["country"].tolist())
    _log(f"fitting corpus stats on {len(names)} records")
    return CorpusStats().fit(names, addrs, countries)


def normalise_split(ss: SourceSet, stats: CorpusStats) -> PreparedSplit:
    def rows(df):
        return zip(
            df["entity_id"].tolist(),
            df["business_name"].tolist(),
            df["business_address"].tolist(),
            df["country"].tolist(),
        )

    s1 = {e: build_record(e, n, a, c, stats) for e, n, a, c in rows(ss.s1)}
    targets: dict[str, NormRecord] = {}
    source_of: dict[str, int] = {}
    for src, df in ((2, ss.s2), (3, ss.s3)):
        for e, n, a, c in rows(df):
            targets[e] = build_record(e, n, a, c, stats)
            source_of[e] = src
    _log(f"normalised {len(s1)} source-1 and {len(targets)} target records")
    return PreparedSplit(s1=s1, targets=targets, source_of=source_of, n_targets=len(targets))


def run_blocking(prep: PreparedSplit, max_candidates: int = 60) -> PreparedSplit:
    _log("building blocking indexes")
    blocker = CountryBlocker(DEFAULT_CHANNELS, max_candidates=max_candidates)
    blocker.fit(list(prep.targets.values()))
    _log("retrieving candidates")
    prep.blocking = blocker.run(list(prep.s1.values()))
    avg = np.mean([len(v) for v in prep.blocking.candidates.values()]) if prep.blocking.candidates else 0
    _log(f"blocking done — {avg:.1f} candidates per source-1 entity")
    return prep


def featurise(prep: PreparedSplit, stats: CorpusStats) -> tuple[np.ndarray, list[Pair]]:
    _log("featurising candidate pairs")
    fz = Featurizer(stats)
    X, pairs = fz.build(prep.s1, prep.targets, prep.blocking.candidates, prep.source_of)
    _log(f"built {X.shape[0]} x {X.shape[1]} feature matrix")
    return X, pairs


# --------------------------------------------------------------------------- #


def scores_by_entity(
    pairs: Sequence[Pair], probs: np.ndarray, all_s1: Sequence[str]
) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {sid: [] for sid in all_s1}
    for p, pr in zip(pairs, probs):
        out.setdefault(p.s1_id, []).append((p.t_id, float(pr)))
    return out


def train_pipeline(
    data_dir: str,
    model_out: str,
    max_candidates: int = 60,
    test_set: SourceSet | None = None,
    holdout_country: str | None = None,
) -> tuple[TwoStageMatcher, CorpusStats, PreparedSplit, list[Pair], np.ndarray, np.ndarray]:
    train_ss = load_sources(data_dir, "train")
    truth = load_ground_truth(data_dir, "train")
    stats = fit_corpus_stats(train_ss, test_set)

    prep = normalise_split(train_ss, stats)
    prep = run_blocking(prep, max_candidates)

    rep = blocking_report(prep.blocking.as_lists(), truth, prep.n_targets)
    _log(
        "blocking quality — "
        + ", ".join(f"{k}={v:.4f}" for k, v in rep.items())
    )
    if rep["pair_recall"] < 0.97:
        _log("WARNING: blocking pair recall below 0.97 — this caps your final score. "
             "Widen top_k on the weakest channel before tuning the model.")

    X, pairs = featurise(prep, stats)
    y = make_labels(pairs, truth)
    _log(f"label balance: {int(y.sum())} positives / {len(y)} pairs")

    holdout: set[str] = set()
    if holdout_country:
        hc = holdout_country.strip().lower()
        holdout = {sid for sid, r in prep.s1.items() if r.country == hc}
        if not holdout:
            _log(f"WARNING: no source-1 entities with country '{holdout_country}'")
        else:
            _log(f"country holdout '{holdout_country}': {len(holdout)} entities excluded from training")

    _log("training two-stage model")
    model = TwoStageMatcher().fit(X, y, pairs, holdout_entities=holdout)
    model.holdout_entities_ = holdout

    os.makedirs(os.path.dirname(os.path.abspath(model_out)) or ".", exist_ok=True)
    model.save(model_out)
    _log(f"model saved to {model_out}")

    return model, stats, prep, pairs, X, y


def predict_pipeline(
    data_dir: str,
    model: TwoStageMatcher,
    stats: CorpusStats,
    out_dir: str,
    decision: dict,
    max_candidates: int = 60,
    enforce_unique: bool = True,
    split: str = "test",
) -> dict[str, list[str]]:
    ss = load_sources(data_dir, split)
    prep = normalise_split(ss, stats)
    prep = run_blocking(prep, max_candidates)

    X, pairs = featurise(prep, stats)
    _log("scoring pairs")
    probs = model.predict_proba(X, pairs)

    all_s1 = list(prep.s1.keys())
    sbe = scores_by_entity(pairs, probs, all_s1)

    _log(f"applying decision rule: {decision.get('strategy', 'expected_f')}")
    preds = apply_strategy(sbe, decision)
    if enforce_unique:
        preds = enforce_unique_assignment(preds, sbe, decision)

    os.makedirs(out_dir, exist_ok=True)
    # candidate_pairs.tsv must be a superset of matching_results.tsv
    write_candidate_pairs(
        os.path.join(out_dir, "candidate_pairs.tsv"),
        ((sid, [t for t, _ in sbe.get(sid, [])]) for sid in all_s1),
    )
    write_matching_results(
        os.path.join(out_dir, "matching_results.tsv"),
        ((sid, preds.get(sid, [])) for sid in all_s1),
    )
    n_pred = sum(len(v) for v in preds.values())
    n_empty = sum(1 for v in preds.values() if not v)
    _log(
        f"wrote {len(all_s1)} rows — {n_pred} predicted matches, "
        f"{n_empty} predicted singletons ({n_empty / max(1, len(all_s1)):.1%})"
    )
    return preds


# --------------------------------------------------------------------------- #


def leave_one_country_out(
    X: np.ndarray,
    y: np.ndarray,
    pairs: Sequence[Pair],
    prep: PreparedSplit,
    min_entities: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Pooled **out-of-country** calibrated predictions.

    For each country, refit the whole two-stage model with that country excluded
    from every training fold *and* from the calibrator, then keep its
    predictions. Pooling these gives you a score distribution produced under the
    same conditions France will face: an unseen country, scored by a model and a
    calibrator that never saw it.

    Selecting the decision rule on this pool rather than on ordinary in-country
    out-of-fold predictions is the single most direct way to protect the French
    third of the test set. It costs one full model fit per country.
    """
    countries = sorted({r.country for r in prep.s1.values() if r.country})
    pooled: dict[str, list[tuple[str, float]]] = {}
    for c in countries:
        ents = {sid for sid, r in prep.s1.items() if r.country == c}
        if len(ents) < min_entities:
            _log(f"  skipping country '{c}' — only {len(ents)} entities")
            continue
        _log(f"  leave-one-country-out: holding out '{c}' ({len(ents)} entities)")
        m = TwoStageMatcher().fit(X, y, pairs, holdout_entities=ents)
        probs = m.oof_calibrated_
        for p, pr in zip(pairs, probs):
            if p.s1_id in ents:
                pooled.setdefault(p.s1_id, []).append((p.t_id, float(pr)))
        for sid in ents:
            pooled.setdefault(sid, [])
    return pooled


def validate_oof(
    model: TwoStageMatcher,
    pairs: Sequence[Pair],
    all_s1: Sequence[str],
    truth: dict[str, set[str]],
    decision: dict,
    enforce_unique: bool = True,
):
    """Score the out-of-fold predictions with the real metric."""
    sbe = scores_by_entity(pairs, model.oof_calibrated_, all_s1)
    preds = apply_strategy(sbe, decision)
    if enforce_unique:
        preds = enforce_unique_assignment(preds, sbe, decision)
    return score_breakdown(preds, truth), sbe, preds
