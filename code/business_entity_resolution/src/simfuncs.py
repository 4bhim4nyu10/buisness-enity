"""String similarity primitives.

Uses rapidfuzz when it is installed (10-50x faster, MIT licensed) and falls back
to dependency-free implementations otherwise, so the pipeline runs anywhere.
"""

from __future__ import annotations

from functools import lru_cache

try:  # pragma: no cover - environment dependent
    from rapidfuzz.distance import JaroWinkler as _JW, Levenshtein as _LV

    _HAVE_RF = True
except Exception:  # pragma: no cover
    _HAVE_RF = False

_MAXLEN = 80  # edit distance is O(n*m); names longer than this are truncated


def _trunc(s: str) -> str:
    return s[:_MAXLEN]


if _HAVE_RF:

    def lev_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(_LV.normalized_similarity(_trunc(a), _trunc(b)))

    def jaro_winkler(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(_JW.normalized_similarity(_trunc(a), _trunc(b)))

else:

    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def lev_ratio(a: str, b: str) -> float:
        a, b = _trunc(a), _trunc(b)
        if not a or not b:
            return 0.0
        m = max(len(a), len(b))
        return 1.0 - _levenshtein(a, b) / m

    def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
        a, b = _trunc(a), _trunc(b)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        la, lb = len(a), len(b)
        window = max(la, lb) // 2 - 1
        if window < 0:
            window = 0
        fa = [False] * la
        fb = [False] * lb
        matches = 0
        for i in range(la):
            lo = max(0, i - window)
            hi = min(i + window + 1, lb)
            for j in range(lo, hi):
                if fb[j] or a[i] != b[j]:
                    continue
                fa[i] = fb[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        k = 0
        transpositions = 0
        for i in range(la):
            if not fa[i]:
                continue
            while not fb[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
        t = transpositions / 2
        jaro = (matches / la + matches / lb + (matches - t) / matches) / 3
        prefix = 0
        for x, y in zip(a[:4], b[:4]):
            if x != y:
                break
            prefix += 1
        return jaro + prefix * p * (1 - jaro)


def jaccard(a: set | frozenset, b: set | frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def containment(a: set | frozenset, b: set | frozenset) -> float:
    """|A n B| / min(|A|,|B|) — robust to one record listing far more tokens."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dice(a: set | frozenset, b: set | frozenset) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def weighted_jaccard(a: set, b: set, weight) -> float:
    """IDF-weighted set overlap. A shared rare token counts far more than a
    shared 'limited'."""
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    wi = sum(weight(t) for t in inter)
    wu = sum(weight(t) for t in (a | b))
    return wi / wu if wu else 0.0


def token_sort_ratio(a_tokens, b_tokens) -> float:
    """Order-insensitive edit similarity — catches word-order transposition."""
    return lev_ratio(" ".join(sorted(a_tokens)), " ".join(sorted(b_tokens)))


@lru_cache(maxsize=200_000)
def _initials(s: str) -> str:
    return "".join(t[0] for t in s.split() if t)


def acronym_score(a_tokens, b_tokens) -> float:
    """1.0 when one side's name is the other side's initials ("IBM" vs
    "International Business Machines")."""
    if not a_tokens or not b_tokens:
        return 0.0
    a_join = "".join(a_tokens)
    b_join = "".join(b_tokens)
    a_ini = "".join(t[0] for t in a_tokens)
    b_ini = "".join(t[0] for t in b_tokens)
    if len(a_tokens) == 1 and len(b_tokens) > 1 and a_join == b_ini:
        return 1.0
    if len(b_tokens) == 1 and len(a_tokens) > 1 and b_join == a_ini:
        return 1.0
    if len(a_tokens) == 1 and len(b_tokens) > 1 and a_join.startswith(b_ini[: len(a_join)]) and len(a_join) >= 2:
        return 0.6
    if len(b_tokens) == 1 and len(a_tokens) > 1 and b_join.startswith(a_ini[: len(b_join)]) and len(b_join) >= 2:
        return 0.6
    return 0.0


def numeric_agreement(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[float, float]:
    """(overlap, conflict) over numeric tokens.

    Numbers are unforgiving evidence: '12' vs '12' supports a match, but
    '12 Main St' vs '48 Main St' is strong evidence *against* one, and that
    asymmetry is exactly what a precision-heavy metric wants.
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0, 0.0
    inter = sa & sb
    overlap = len(inter) / min(len(sa), len(sb))
    conflict = 0.0 if inter else 1.0
    return overlap, conflict


def prefix_match_len(a: str, b: str, cap: int = 6) -> float:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
        if n >= cap:
            break
    return n / cap
