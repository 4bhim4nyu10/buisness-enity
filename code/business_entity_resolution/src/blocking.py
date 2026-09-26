"""
Candidate generation (blocking).

Blocking sets the ceiling on everything downstream: a true match that never
becomes a candidate can never be recovered by the model. So the objective here
is **recall first**, with reduction ratio as the secondary constraint.

We run several complementary recall channels and take their union:

  A. char 3-gram similarity on the core name   -> typos, transpositions, spacing
  B. word-token similarity on name + address   -> reordering, partial addresses
  C. rare-token inverted index (name)          -> distinctive words ("sundaram")
  D. rare-token inverted index (address)       -> distinctive street/locality
  E. phonetic key on the core name             -> transliteration variants
  F. postal / house-number key + name initial  -> same building, mangled name

Each channel is an inverted index with **posting-list pruning**: any feature
whose posting list covers more than `max_df_frac` of the corpus is skipped
entirely. That single trick is what keeps this O(corpus) instead of O(n^2) —
common trigrams like "ion" or words like "limited" never get traversed, and
they carry no discriminative signal anyway.

Everything is done inside a country bucket, with a global fallback bucket so
that an unseen country label (France) or a blank one still retrieves.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .normalize import NormRecord, char_ngrams


# --------------------------------------------------------------------------- #
# Inverted index with top-K weighted-overlap retrieval
# --------------------------------------------------------------------------- #


class InvertedIndex:
    """Sparse cosine-ish retrieval without materialising any dense matrix.

    Scores a query against every document that shares at least one *selective*
    feature with it. Score = sum of idf(f)^2 over shared features, divided by
    the product of the vectors' L2 norms (so it really is a cosine over an
    idf-weighted binary bag).
    """

    def __init__(self, max_df_frac: float = 0.02, min_postings_cap: int = 200):
        self.postings: dict[str, np.ndarray] = {}
        self.norms: np.ndarray = np.zeros(0, dtype=np.float32)
        self.n_docs = 0
        self.idf: dict[str, float] = {}
        self.max_postings = 0
        self.min_postings_cap = min_postings_cap
        self.max_df_frac = max_df_frac
        self.doc_ids: list[str] = []

    def fit(self, docs: Sequence[Iterable[str]], doc_ids: Sequence[str]) -> "InvertedIndex":
        self.n_docs = len(docs)
        self.doc_ids = list(doc_ids)
        df: dict[str, int] = defaultdict(int)
        feats: list[list[str]] = []
        for d in docs:
            fs = list(dict.fromkeys(d))  # dedupe, keep order
            feats.append(fs)
            for f in fs:
                df[f] += 1

        self.idf = {
            f: math.log((self.n_docs + 1.0) / (c + 1.0)) + 1.0 for f, c in df.items()
        }
        self.max_postings = max(
            self.min_postings_cap, int(self.n_docs * self.max_df_frac)
        )

        postings: dict[str, list[int]] = defaultdict(list)
        norms = np.zeros(self.n_docs, dtype=np.float32)
        for i, fs in enumerate(feats):
            acc = 0.0
            for f in fs:
                if df[f] > self.max_postings:
                    continue
                w = self.idf[f]
                postings[f].append(i)
                acc += w * w
            norms[i] = math.sqrt(acc) if acc > 0 else 1.0
        self.postings = {f: np.asarray(v, dtype=np.int32) for f, v in postings.items()}
        self.norms = norms
        return self

    def query(self, feats: Iterable[str], top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        qnorm = 0.0
        for f in dict.fromkeys(feats):
            w = self.idf.get(f)
            if w is None:
                continue
            qnorm += w * w
            post = self.postings.get(f)
            if post is None:
                continue
            ww = w * w
            for idx in post:
                scores[idx] += ww
        if not scores:
            return []
        qnorm = math.sqrt(qnorm) if qnorm > 0 else 1.0

        idxs = np.fromiter(scores.keys(), dtype=np.int32, count=len(scores))
        vals = np.fromiter(scores.values(), dtype=np.float32, count=len(scores))
        vals /= (qnorm * self.norms[idxs])
        if len(idxs) > top_k:
            sel = np.argpartition(-vals, top_k)[:top_k]
            idxs, vals = idxs[sel], vals[sel]
        order = np.argsort(-vals)
        return [(int(idxs[i]), float(vals[i])) for i in order]


# --------------------------------------------------------------------------- #
# Feature extractors, one per recall channel
# --------------------------------------------------------------------------- #


def f_name_3gram(r: NormRecord) -> list[str]:
    return [f"n3:{g}" for g in r.name_3grams]


def f_word(r: NormRecord) -> list[str]:
    return [f"w:{t}" for t in r.core_tokens] + [f"a:{t}" for t in r.addr_core_tokens]


def f_name_token(r: NormRecord) -> list[str]:
    return [f"t:{t}" for t in r.core_tokens]


def f_addr_token(r: NormRecord) -> list[str]:
    return [f"at:{t}" for t in r.addr_core_tokens]


def f_phonetic(r: NormRecord) -> list[str]:
    return [f"p:{p}" for p in r.name_phon.split()]


def f_struct(r: NormRecord) -> list[str]:
    """Structural keys: postal code, house numbers, name-initials."""
    out: list[str] = []
    ini = r.acronym[:4]
    if r.postal:
        out.append(f"pc:{r.postal}")
        if ini:
            out.append(f"pc+i:{r.postal}|{ini}")
    for num in set(r.addr_nums):
        if len(num) <= 5:
            out.append(f"hn:{num}|{ini[:2]}")
    if len(r.core_tokens) >= 1:
        longest = max(r.core_tokens, key=len)
        if len(longest) >= 5:
            out.append(f"lt:{longest}")
            out.append(f"ltp:{longest[:5]}")
    return out


def f_addr_3gram(r: NormRecord) -> list[str]:
    return [f"a3:{g}" for g in r.addr_3grams]


@dataclass
class Channel:
    name: str
    extractor: Callable[[NormRecord], list[str]]
    top_k: int
    max_df_frac: float = 0.02


DEFAULT_CHANNELS: list[Channel] = [
    Channel("name3gram", f_name_3gram, top_k=30, max_df_frac=0.05),
    Channel("word", f_word, top_k=30, max_df_frac=0.05),
    Channel("nametok", f_name_token, top_k=20, max_df_frac=0.02),
    Channel("addrtok", f_addr_token, top_k=15, max_df_frac=0.02),
    Channel("phonetic", f_phonetic, top_k=15, max_df_frac=0.02),
    Channel("struct", f_struct, top_k=15, max_df_frac=0.01),
    Channel("addr3gram", f_addr_3gram, top_k=10, max_df_frac=0.05),
]


# --------------------------------------------------------------------------- #
# Blocker
# --------------------------------------------------------------------------- #


@dataclass
class BlockingResult:
    # source1 entity_id -> list of (target entity_id, dict of channel scores)
    candidates: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    def as_lists(self) -> dict[str, list[str]]:
        return {k: list(v.keys()) for k, v in self.candidates.items()}


class CountryBlocker:
    """Builds one index set per country bucket, plus a global bucket."""

    def __init__(
        self,
        channels: Sequence[Channel] | None = None,
        max_candidates: int = 60,
        use_global_fallback: bool = True,
        min_bucket_size: int = 50,
    ):
        self.channels = list(channels or DEFAULT_CHANNELS)
        self.max_candidates = max_candidates
        self.use_global_fallback = use_global_fallback
        self.min_bucket_size = min_bucket_size
        self._buckets: dict[str, dict[str, InvertedIndex]] = {}
        self._bucket_records: dict[str, list[NormRecord]] = {}

    def fit(self, targets: Sequence[NormRecord]) -> "CountryBlocker":
        buckets: dict[str, list[NormRecord]] = defaultdict(list)
        for r in targets:
            buckets[r.country].append(r)
        # a small country bucket is unreliable -> merge it into GLOBAL only
        buckets["__GLOBAL__"] = list(targets)

        for cname, recs in buckets.items():
            if cname != "__GLOBAL__" and len(recs) < self.min_bucket_size:
                continue
            idxs: dict[str, InvertedIndex] = {}
            for ch in self.channels:
                ii = InvertedIndex(max_df_frac=ch.max_df_frac)
                ii.fit([ch.extractor(r) for r in recs], [r.entity_id for r in recs])
                idxs[ch.name] = ii
            self._buckets[cname] = idxs
            self._bucket_records[cname] = recs
        return self

    def _query_bucket(self, bucket: str, rec: NormRecord, out: dict[str, dict[str, float]]) -> None:
        idxs = self._buckets.get(bucket)
        if not idxs:
            return
        recs = self._bucket_records[bucket]
        for ch in self.channels:
            ii = idxs[ch.name]
            for doc_i, score in ii.query(ch.extractor(rec), ch.top_k):
                eid = recs[doc_i].entity_id
                slot = out.setdefault(eid, {})
                if score > slot.get(ch.name, 0.0):
                    slot[ch.name] = score

    def candidates_for(self, rec: NormRecord) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if rec.country in self._buckets:
            self._query_bucket(rec.country, rec, out)
        if self.use_global_fallback and (
            rec.country not in self._buckets or len(out) < self.max_candidates // 2
        ):
            self._query_bucket("__GLOBAL__", rec, out)

        if len(out) > self.max_candidates:
            ranked = sorted(out.items(), key=lambda kv: -_prescore(kv[1]))
            out = dict(ranked[: self.max_candidates])
        return out

    def run(self, queries: Sequence[NormRecord], progress_every: int = 5000) -> BlockingResult:
        res = BlockingResult()
        for i, q in enumerate(queries):
            res.candidates[q.entity_id] = self.candidates_for(q)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  blocking {i + 1}/{len(queries)}", flush=True)
        return res


_CH_WEIGHTS = {
    "name3gram": 1.0,
    "word": 0.9,
    "nametok": 1.0,
    "addrtok": 0.5,
    "phonetic": 0.6,
    "struct": 0.7,
    "addr3gram": 0.4,
}


def _prescore(chan_scores: dict[str, float]) -> float:
    """Cheap fusion used only to trim an over-full candidate list."""
    s = 0.0
    for k, v in chan_scores.items():
        s += _CH_WEIGHTS.get(k, 0.5) * v
    return s + 0.15 * len(chan_scores)


# --------------------------------------------------------------------------- #
# Diagnostics — run these every time you touch a channel
# --------------------------------------------------------------------------- #


def blocking_report(
    cands: dict[str, list[str]],
    truth: dict[str, set[str]],
    n_targets: int,
) -> dict[str, float]:
    """Recall ceiling and reduction ratio.

    `pair_recall`   - fraction of true pairs that survived blocking.
    `entity_recall` - fraction of entities whose *entire* truth set survived
                      (this is the one that matters for a macro metric).
    `reduction`     - 1 - kept_pairs / all_possible_pairs.
    """
    tp = fn = 0
    full = 0
    n_ent = 0
    kept = 0
    for sid, gold in truth.items():
        got = set(cands.get(sid, []))
        kept += len(got)
        n_ent += 1
        if not gold:
            full += 1
            continue
        hit = len(gold & got)
        tp += hit
        fn += len(gold) - hit
        if hit == len(gold):
            full += 1
    total_pairs = max(1, n_ent * n_targets)
    return {
        "pair_recall": tp / max(1, tp + fn),
        "entity_recall": full / max(1, n_ent),
        "avg_candidates": kept / max(1, n_ent),
        "reduction_ratio": 1.0 - kept / total_pairs,
    }
