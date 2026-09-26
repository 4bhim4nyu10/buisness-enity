"""Exploratory analysis. Run this FIRST, before writing a line of model code.

Everything downstream depends on facts you can only get from the data:

  * What fraction of Source-1 entities are singletons? That fraction is the
    share of your score decided purely by knowing when to predict nothing.
  * How large are true match sets? If nearly all are 0 or 1, the decision layer
    simplifies enormously.
  * Is a Source-2/3 record ever matched to two different Source-1 entities? If
    not, the unique-assignment constraint is safe to enforce.
  * How different are the noise patterns between US and India, and between
    Source 2 and Source 3?

Usage:  python -m src.analyze --data-dir dataset
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from .dataio import load_ground_truth, load_sources
from .normalize import CorpusStats, build_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    args = ap.parse_args()

    train = load_sources(args.data_dir, "train")
    truth = load_ground_truth(args.data_dir, "train")

    print("=" * 70)
    print("ROW COUNTS")
    print(f"  source1 {len(train.s1):>8}   source2 {len(train.s2):>8}   source3 {len(train.s3):>8}")
    try:
        test = load_sources(args.data_dir, "test")
        print(f"  TEST: s1 {len(test.s1):>8}   s2 {len(test.s2):>8}   s3 {len(test.s3):>8}")
    except FileNotFoundError:
        test = None
        print("  (no test split found)")

    print("\nCOUNTRY DISTRIBUTION")
    for name, ss in (("train", train), ("test", test)):
        if ss is None:
            continue
        c = Counter(ss.all_records()["country"].str.strip())
        print(f"  {name}: {dict(c.most_common(10))}")

    print("\nMATCH-SET SIZE  (this drives the whole decision layer)")
    sizes = Counter(len(v) for v in truth.values())
    total = sum(sizes.values())
    for k in sorted(sizes):
        print(f"  {k:>3} matches : {sizes[k]:>8}  ({sizes[k] / total:6.2%})")
    singles = sizes.get(0, 0) / total if total else 0
    print(f"\n  >> singletons are {singles:.1%} of entities.")
    print(f"     A model that predicts nothing for everyone scores {singles:.4f}.")
    print("     That is your floor — beat it by a lot or something is wrong.")

    print("\nPER-SOURCE MATCHES")
    s2 = sum(1 for v in truth.values() for x in v if x.startswith("S2"))
    s3 = sum(1 for v in truth.values() for x in v if x.startswith("S3"))
    print(f"  from source2: {s2}   from source3: {s3}")
    both = sum(1 for v in truth.values() if any(x.startswith("S2") for x in v) and any(x.startswith("S3") for x in v))
    print(f"  entities matching in BOTH sources: {both}")

    print("\nUNIQUE-ASSIGNMENT CHECK")
    owner: dict[str, int] = Counter()
    for v in truth.values():
        owner.update(v)
    shared = [k for k, c in owner.items() if c > 1]
    print(f"  target records claimed by >1 source-1 entity: {len(shared)}")
    if shared:
        print("  -> set enforce_unique=False in the decision layer")
    else:
        print("  -> unique-assignment constraint is SAFE to enforce")

    print("\nFIELD COMPLETENESS")
    for nm, df in (("s1", train.s1), ("s2", train.s2), ("s3", train.s3)):
        empty_addr = (df["business_address"].str.strip() == "").mean()
        empty_ctry = (df["country"].str.strip() == "").mean()
        ln = df["business_name"].str.len()
        print(f"  {nm}: empty address {empty_addr:.2%}, empty country {empty_ctry:.2%}, "
              f"name len mean {ln.mean():.1f} p95 {np.percentile(ln, 95):.0f}")

    print("\nNOISE SAMPLES (true pairs — read a dozen of these by hand)")
    stats = CorpusStats().fit(
        train.all_records()["business_name"].tolist(),
        train.all_records()["business_address"].tolist(),
        train.all_records()["country"].tolist(),
    )
    print("  top corpus-generic NAME tokens (auto-learned, no hardcoding):")
    print("   ", [t for t, _ in stats.name_df.most_common(25)])
    print("  top corpus-generic ADDRESS tokens:")
    print("   ", [t for t, _ in stats.addr_df.most_common(25)])

    idx1 = {r["entity_id"]: r for _, r in train.s1.iterrows()}
    idx_t = {}
    for df in (train.s2, train.s3):
        for _, r in df.iterrows():
            idx_t[r["entity_id"]] = r
    shown = 0
    for sid, gold in truth.items():
        if not gold or shown >= 12:
            continue
        a = idx1.get(sid)
        if a is None:
            continue
        for g in list(gold)[:1]:
            b = idx_t.get(g)
            if b is None:
                continue
            print(f"\n  [{a['country']}] {a['business_name']}")
            print(f"        {a['business_address']}")
            print(f"     ~= {b['business_name']}")
            print(f"        {b['business_address']}")
            shown += 1
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
