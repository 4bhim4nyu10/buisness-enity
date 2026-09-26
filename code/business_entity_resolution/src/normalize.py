"""
Text normalisation for business names and addresses.

Design principle: as little hard-coded, country-specific knowledge as possible.
The test set contains a country (France) that never appears in training, so any
rule keyed on "US" or "India" is a liability. Instead we:

  1. apply only *universal* string hygiene (unicode folding, case, punctuation),
  2. apply a small set of *orthographic* abbreviation rules that are about
     writing systems rather than about a particular country,
  3. learn the "generic / uninformative token" vocabulary (legal suffixes,
     industry words, street-type words) **from the corpus itself** at runtime.

Point 3 is what makes France work for free: `SARL`, `SAS`, `RUE`, `AVENUE` will
be high-document-frequency tokens in the French part of the test corpus, so they
get demoted automatically without anybody writing them down.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Universal orthographic rules.
# These are about how abbreviations are written, not about any one country.
# Keep this list SHORT and boring; the corpus-driven stats do the heavy lifting.
# --------------------------------------------------------------------------- #

_AMP = re.compile(r"\s*&\s*")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_ALNUM_SPLIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")

# Abbreviation -> expansion. Applied as whole tokens only.
ABBREV = {
    # corporate forms (written-form variants, present across many countries)
    "corp": "corporation",
    "co": "company",
    "cos": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "ltda": "limited",
    "lmtd": "limited",
    "pvt": "private",
    "pte": "private",
    "prvt": "private",
    "llp": "llp",
    "llc": "llc",
    "intl": "international",
    "int": "international",
    "natl": "national",
    "mfg": "manufacturing",
    "mfrs": "manufacturers",
    "ind": "industries",
    "inds": "industries",
    "ent": "enterprises",
    "svc": "services",
    "svcs": "services",
    "serv": "services",
    "tech": "technologies",
    "techno": "technologies",
    "sols": "solutions",
    "grp": "group",
    "bros": "brothers",
    "assoc": "associates",
    "assocs": "associates",
    "mkt": "market",
    "dept": "department",
    "univ": "university",
    "hosp": "hospital",
    "rest": "restaurant",
    # street types
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "expy": "expressway",
    "pkwy": "parkway",
    "ct": "court",
    "cir": "circle",
    "sq": "square",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "rm": "room",
    "opp": "opposite",
    "nr": "near",
    "flr": "floor",
    "gr": "ground",
    "gf": "ground floor",
    "no": "number",
    "ph": "phase",
    "sec": "sector",
    "mg": "mahatma gandhi",
    # directions
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# Tokens that are *always* noise regardless of corpus (pure filler).
STOP_TOKENS = {"the", "and", "of", "at", "in", "on", "for", "de", "du", "des", "la", "le", "les", "el"}


def strip_accents(s: str) -> str:
    """NFKD fold then drop combining marks. Handles é->e, ß->ss-ish, ı->i."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def basic_clean(s: object) -> str:
    """Lowercase, fold accents, split letter/digit runs, strip punctuation."""
    if s is None:
        return ""
    s = str(s)
    if s.lower() in {"nan", "none", "null"}:
        return ""
    s = strip_accents(s).lower()
    s = _AMP.sub(" and ", s)
    s = s.replace("'", "").replace("`", "").replace("’", "")
    s = _PUNCT.sub(" ", s)
    s = _ALNUM_SPLIT.sub(" ", s)          # "flat12" -> "flat 12"
    return _WS.sub(" ", s).strip()


def expand_tokens(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        rep = ABBREV.get(t)
        if rep:
            out.extend(rep.split())
        else:
            out.append(t)
    return out


def tokenize(s: str) -> list[str]:
    """Clean + expand abbreviations + drop pure filler."""
    toks = basic_clean(s).split()
    toks = expand_tokens(toks)
    return [t for t in toks if t and t not in STOP_TOKENS]


# --------------------------------------------------------------------------- #
# Lightweight phonetic key (no external deps).
# A trimmed Metaphone: good enough to bridge transliteration variants like
# "Shree"/"Shri", "Lakshmi"/"Laxmi", "Kumar"/"Kumaar".
# --------------------------------------------------------------------------- #

_VOWELS = set("aeiou")
_PHON_SUBS = [
    ("ph", "f"), ("gh", "f"), ("ck", "k"), ("cq", "k"), ("qu", "k"), ("q", "k"),
    ("sch", "sk"), ("sh", "s"), ("ch", "s"), ("th", "t"), ("dh", "d"),
    ("bh", "b"), ("gh", "g"), ("jh", "j"), ("kh", "k"), ("ph", "f"),
    ("wh", "w"), ("x", "ks"), ("z", "s"), ("c", "k"), ("y", "i"),
    ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i"), ("uu", "u"),
    ("v", "w"),
]


def phonetic(token: str) -> str:
    """Collapse a token to a coarse consonant skeleton."""
    t = basic_clean(token)
    if not t:
        return ""
    for a, b in _PHON_SUBS:
        t = t.replace(a, b)
    first = t[0]
    body = "".join(ch for ch in t[1:] if ch not in _VOWELS)
    # squeeze repeats
    squeezed = []
    for ch in first + body:
        if not squeezed or squeezed[-1] != ch:
            squeezed.append(ch)
    return "".join(squeezed)[:8]


def phonetic_key(tokens: Iterable[str]) -> str:
    return " ".join(p for p in (phonetic(t) for t in tokens) if p)


# --------------------------------------------------------------------------- #
# Corpus-driven generic-token mining
# --------------------------------------------------------------------------- #


@dataclass
class CorpusStats:
    """Document frequencies learned from every record we can see.

    Fit this on train + test records together (transductive use of the provided
    data, which the rules allow — no external lookup). That way French legal
    suffixes and street words get demoted even though no French row was ever
    labelled.
    """

    name_df: Counter = field(default_factory=Counter)
    addr_df: Counter = field(default_factory=Counter)
    n_docs: int = 0
    generic_name_tokens: set[str] = field(default_factory=set)
    generic_addr_tokens: set[str] = field(default_factory=set)
    # per-country stats so "restaurant" can be generic in one market only
    by_country: dict[str, Counter] = field(default_factory=dict)

    def fit(
        self,
        names: Iterable[str],
        addresses: Iterable[str],
        countries: Iterable[str] | None = None,
        name_generic_frac: float = 0.01,
        addr_generic_frac: float = 0.02,
    ) -> "CorpusStats":
        names = list(names)
        addresses = list(addresses)
        countries = list(countries) if countries is not None else [""] * len(names)
        self.n_docs = len(names)

        for nm, ad, ct in zip(names, addresses, countries):
            ntok = set(tokenize(nm))
            atok = set(tokenize(ad))
            self.name_df.update(ntok)
            self.addr_df.update(atok)
            c = basic_clean(ct)
            self.by_country.setdefault(c, Counter()).update(ntok)

        if self.n_docs:
            nthr = max(20, int(self.n_docs * name_generic_frac))
            athr = max(20, int(self.n_docs * addr_generic_frac))
            self.generic_name_tokens = {t for t, c in self.name_df.items() if c >= nthr}
            self.generic_addr_tokens = {t for t, c in self.addr_df.items() if c >= athr}
        return self

    def idf(self, token: str, field_: str = "name") -> float:
        """Smoothed IDF. Rare tokens ('kalyanaraman') get high weight; generic
        ones ('limited', 'sarl', 'road') get almost none."""
        import math

        df = self.name_df.get(token, 0) if field_ == "name" else self.addr_df.get(token, 0)
        return math.log((self.n_docs + 1.0) / (df + 1.0))

    def is_generic(self, token: str, field_: str = "name") -> bool:
        s = self.generic_name_tokens if field_ == "name" else self.generic_addr_tokens
        return token in s


# --------------------------------------------------------------------------- #
# Record-level normalised view
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class NormRecord:
    entity_id: str
    country: str
    name_norm: str            # full normalised name
    name_tokens: tuple[str, ...]
    core_tokens: tuple[str, ...]      # name minus corpus-generic tokens
    core_name: str
    name_phon: str
    acronym: str              # first letters of core tokens, e.g. "ibm"
    addr_norm: str
    addr_tokens: tuple[str, ...]
    addr_core_tokens: tuple[str, ...]
    addr_nums: tuple[str, ...]        # all digit runs in the address
    postal: str                       # best postal-code guess ("" if none)
    name_3grams: frozenset[str]
    addr_3grams: frozenset[str]


def char_ngrams(s: str, n: int = 3) -> frozenset[str]:
    s = f" {s} "
    if len(s) < n:
        return frozenset({s})
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1))


_POSTAL_RE = re.compile(r"\b\d{4,6}\b")


def build_record(
    entity_id: str,
    name: object,
    address: object,
    country: object,
    stats: CorpusStats | None,
) -> NormRecord:
    ntok = tokenize(name)
    atok = tokenize(address)

    if stats is not None:
        core = [t for t in ntok if not stats.is_generic(t, "name")] or ntok
        acore = [t for t in atok if not stats.is_generic(t, "addr")] or atok
    else:
        core, acore = ntok, atok

    raw_addr = basic_clean(address)
    nums = tuple(_DIGITS.findall(raw_addr))
    postal_candidates = _POSTAL_RE.findall(raw_addr)
    # postal codes are usually the longest / last numeric block
    postal = ""
    if postal_candidates:
        postal = max(postal_candidates, key=lambda x: (len(x), raw_addr.rfind(x)))

    core_name = " ".join(core)
    return NormRecord(
        entity_id=entity_id,
        country=basic_clean(country),
        name_norm=" ".join(ntok),
        name_tokens=tuple(ntok),
        core_tokens=tuple(core),
        core_name=core_name,
        name_phon=phonetic_key(core),
        acronym="".join(t[0] for t in core if t),
        addr_norm=" ".join(atok),
        addr_tokens=tuple(atok),
        addr_core_tokens=tuple(acore),
        addr_nums=nums,
        postal=postal,
        name_3grams=char_ngrams(core_name or " ".join(ntok)),
        addr_3grams=char_ngrams(" ".join(atok)),
    )
