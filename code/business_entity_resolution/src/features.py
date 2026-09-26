"""
Pairwise feature engineering.

Two families of features, and the second is the one most teams forget:

  * **Intrinsic** — how similar are these two records? (string, token, numeric,
    structural similarity)
  * **Relational / competitive** — how does this pair compare to the *other*
    pairs it is competing with? Source 1 is a **deduplicated** reference, so a
    Source-2 record can legitimately belong to at most one Source-1 entity.
    That turns matching into a competition, and features like "is this the best
    Source-1 candidate for this Source-2 record?" and "how far behind the top
    candidate is this one?" carry enormous precision signal.

Relational features need a first pass of intrinsic scores, so featurisation is
two-stage: `base_matrix()` then `add_relational()`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .blocking import _CH_WEIGHTS, _prescore
from .normalize import CorpusStats, NormRecord
from .simfuncs import (
    acronym_score,
    containment,
    dice,
    jaccard,
    jaro_winkler,
    lev_ratio,
    numeric_agreement,
    prefix_match_len,
    token_sort_ratio,
    weighted_jaccard,
)

BASE_FEATURES = [
    # --- name, token level -------------------------------------------------
    "name_tok_jaccard",
    "core_tok_jaccard",
    "core_tok_containment",
    "core_tok_dice",
    "core_tok_widf_jaccard",
    "max_shared_idf",
    "sum_unmatched_idf_a",
    "sum_unmatched_idf_b",
    "n_shared_core_tokens",
    "first_token_eq",
    "first_token_sim",
    # --- name, character level --------------------------------------------
    "name_3gram_jaccard",
    "core_lev",
    "core_jw",
    "full_name_lev",
    "token_sort_lev",
    "acronym",
    "name_len_ratio",
    "name_prefix",
    # --- phonetics ---------------------------------------------------------
    "phon_jaccard",
    "phon_lev",
    # --- address -----------------------------------------------------------
    "addr_tok_jaccard",
    "addr_tok_containment",
    "addr_widf_jaccard",
    "addr_3gram_jaccard",
    "addr_lev",
    "postal_eq",
    "postal_prefix",
    "num_overlap",
    "num_conflict",
    "addr_missing_a",
    "addr_missing_b",
    "addr_len_ratio",
    # --- context -----------------------------------------------------------
    "country_eq",
    "country_known",
    "is_source3",
    "n_channels_hit",
    "prescore",
]
CHANNEL_FEATURES = [f"ch_{k}" for k in _CH_WEIGHTS]

RELATIONAL_FEATURES = [
    "n_cands_for_s1",
    "rank_in_s1",
    "score_over_best_s1",
    "score_minus_second_s1",
    "score_zscore_s1",
    "n_s1_for_target",
    "rank_in_target",
    "score_over_best_target",
    "mutual_best",
    "s1_best_is_unique",
    "same_source_rank",
]

FEATURE_NAMES = BASE_FEATURES + CHANNEL_FEATURES + RELATIONAL_FEATURES


@dataclass
class Pair:
    s1_id: str
    t_id: str
    source: int
    chan: dict[str, float]


class Featurizer:
    def __init__(self, stats: CorpusStats):
        self.stats = stats

    # ------------------------------------------------------------------ #

    def _idf_name(self, t: str) -> float:
        return self.stats.idf(t, "name")

    def _idf_addr(self, t: str) -> float:
        return self.stats.idf(t, "addr")

    def base_vector(self, a: NormRecord, b: NormRecord, source: int, chan: dict[str, float]) -> list[float]:
        an, bn = set(a.name_tokens), set(b.name_tokens)
        ac, bc = set(a.core_tokens), set(b.core_tokens)
        aa, ba = set(a.addr_tokens), set(b.addr_tokens)
        shared = ac & bc

        idfs_shared = [self._idf_name(t) for t in shared]
        unmatched_a = [self._idf_name(t) for t in (ac - bc)]
        unmatched_b = [self._idf_name(t) for t in (bc - ac)]
        denom_a = sum(self._idf_name(t) for t in ac) or 1.0
        denom_b = sum(self._idf_name(t) for t in bc) or 1.0

        ap = set(a.name_phon.split())
        bp = set(b.name_phon.split())

        num_ov, num_cf = numeric_agreement(a.addr_nums, b.addr_nums)
        la, lb = len(a.core_name) or 1, len(b.core_name) or 1
        ala, alb = len(a.addr_norm) or 1, len(b.addr_norm) or 1

        af = a.core_tokens[0] if a.core_tokens else ""
        bf = b.core_tokens[0] if b.core_tokens else ""

        vec = [
            jaccard(an, bn),
            jaccard(ac, bc),
            containment(ac, bc),
            dice(ac, bc),
            weighted_jaccard(ac, bc, self._idf_name),
            max(idfs_shared) if idfs_shared else 0.0,
            sum(unmatched_a) / denom_a,
            sum(unmatched_b) / denom_b,
            float(len(shared)),
            float(af == bf and af != ""),
            jaro_winkler(af, bf),
            jaccard(a.name_3grams, b.name_3grams),
            lev_ratio(a.core_name, b.core_name),
            jaro_winkler(a.core_name, b.core_name),
            lev_ratio(a.name_norm, b.name_norm),
            token_sort_lev_safe(a.core_tokens, b.core_tokens),
            acronym_score(a.core_tokens, b.core_tokens),
            min(la, lb) / max(la, lb),
            prefix_match_len(a.core_name, b.core_name),
            jaccard(ap, bp),
            lev_ratio(a.name_phon, b.name_phon),
            jaccard(aa, ba),
            containment(aa, ba),
            weighted_jaccard(aa, ba, self._idf_addr),
            jaccard(a.addr_3grams, b.addr_3grams),
            lev_ratio(a.addr_norm, b.addr_norm),
            float(bool(a.postal) and a.postal == b.postal),
            prefix_match_len(a.postal, b.postal, cap=6) if (a.postal and b.postal) else 0.0,
            num_ov,
            num_cf,
            float(not a.addr_norm),
            float(not b.addr_norm),
            min(ala, alb) / max(ala, alb),
            float(a.country == b.country),
            float(bool(a.country) and bool(b.country)),
            float(source == 3),
            float(len(chan)),
            _prescore(chan),
        ]
        vec.extend(chan.get(k, 0.0) for k in _CH_WEIGHTS)
        return vec

    # ------------------------------------------------------------------ #

    def build(
        self,
        s1_records: dict[str, NormRecord],
        target_records: dict[str, NormRecord],
        candidates: dict[str, dict[str, dict[str, float]]],
        source_of: dict[str, int],
        progress_every: int = 20_000,
    ) -> tuple[np.ndarray, list[Pair]]:
        """Intrinsic features for every candidate pair."""
        rows: list[list[float]] = []
        pairs: list[Pair] = []
        n = 0
        for s1_id, cands in candidates.items():
            a = s1_records.get(s1_id)
            if a is None:
                continue
            for t_id, chan in cands.items():
                b = target_records.get(t_id)
                if b is None:
                    continue
                src = source_of.get(t_id, 2)
                rows.append(self.base_vector(a, b, src, chan))
                pairs.append(Pair(s1_id, t_id, src, chan))
                n += 1
                if progress_every and n % progress_every == 0:
                    print(f"  featurising {n} pairs", flush=True)
        X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(BASE_FEATURES) + len(CHANNEL_FEATURES)), np.float32)
        return X, pairs


def token_sort_lev_safe(a_tokens, b_tokens) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    return token_sort_ratio(a_tokens, b_tokens)


# --------------------------------------------------------------------------- #
# Relational block
# --------------------------------------------------------------------------- #


def add_relational(X: np.ndarray, pairs: Sequence[Pair], base_score: np.ndarray) -> np.ndarray:
    """Append competition features derived from a first-pass score.

    `base_score` is any monotone proxy for match probability. In training we use
    an out-of-fold prediction from a first-stage model; at inference we use the
    first-stage model's prediction. Using the *same* construction in both places
    is essential — mismatched relational features are a classic silent leak.
    """
    n = len(pairs)
    by_s1: dict[str, list[int]] = defaultdict(list)
    by_t: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by_s1[p.s1_id].append(i)
        by_t[p.t_id].append(i)

    R = np.zeros((n, len(RELATIONAL_FEATURES)), dtype=np.float32)

    best_for_s1: dict[str, int] = {}
    for s1, idxs in by_s1.items():
        sc = base_score[idxs]
        order = np.argsort(-sc)
        ranked = [idxs[j] for j in order]
        best_for_s1[s1] = ranked[0]
        top = sc[order[0]]
        second = sc[order[1]] if len(order) > 1 else 0.0
        mu, sd = float(sc.mean()), float(sc.std()) or 1e-6
        # rank within the same source, so S2 and S3 don't crowd each other out
        per_src: dict[int, int] = defaultdict(int)
        for rank, i in enumerate(ranked):
            R[i, 0] = len(idxs)
            R[i, 1] = rank
            R[i, 2] = base_score[i] / top if top > 0 else 0.0
            R[i, 3] = base_score[i] - second
            R[i, 4] = (base_score[i] - mu) / sd
            R[i, 10] = per_src[pairs[i].source]
            per_src[pairs[i].source] += 1
        R[np.asarray(idxs), 9] = float(top - second > 0.15)

    best_for_t: dict[str, int] = {}
    for t, idxs in by_t.items():
        sc = base_score[idxs]
        order = np.argsort(-sc)
        ranked = [idxs[j] for j in order]
        best_for_t[t] = ranked[0]
        top = sc[order[0]]
        for rank, i in enumerate(ranked):
            R[i, 5] = len(idxs)
            R[i, 6] = rank
            R[i, 7] = base_score[i] / top if top > 0 else 0.0

    for i, p in enumerate(pairs):
        R[i, 8] = float(best_for_s1.get(p.s1_id) == i and best_for_t.get(p.t_id) == i)

    return np.hstack([X, R])
