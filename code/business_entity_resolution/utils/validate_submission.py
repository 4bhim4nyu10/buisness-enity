"""Validates matching_results.tsv and candidate_pairs.tsv against the rules in
the challenge problem statement (stdlib only, no dependencies).

Prints PASS (exit 0) when the files are safe to submit, or a numbered list of
issues to fix (exit 1). It only reads the output files and the test source
files; it does not compute the F_0.5 score. It also prints two diagnostics
that aren't required for validation but are useful early warnings: the
fraction of entities predicted as singletons, and the size distribution of
predicted sets — both signal whether the decision rule has drifted.

    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys


def read_id_list(path: str) -> tuple[list[str], dict[str, list[str]], list[str]]:
    issues: list[str] = []
    order: list[str] = []
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(rdr, None)
        if not header or len(header) != 2:
            issues.append(f"{path}: header must have exactly 2 tab-separated columns, got {header}")
            return order, out, issues
        if header[0] != "source1_entity_id":
            issues.append(f"{path}: first column must be 'source1_entity_id', got '{header[0]}'")
        for ln, row in enumerate(rdr, start=2):
            if len(row) == 1:
                row = [row[0], ""]
            if len(row) != 2:
                issues.append(f"{path}:{ln}: expected 2 columns, got {len(row)}")
                continue
            sid, ids = row[0].strip(), row[1].strip()
            lst = [x.strip() for x in ids.split(",") if x.strip()] if ids else []
            if sid in out:
                issues.append(f"{path}:{ln}: duplicate source1_entity_id '{sid}'")
            if len(set(lst)) != len(lst):
                dup = [k for k, c in collections.Counter(lst).items() if c > 1]
                issues.append(f"{path}:{ln}: duplicate ids within the list for '{sid}': {dup}")
            order.append(sid)
            out[sid] = lst
    return order, out, issues


def read_ids(path: str) -> set[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(rdr, None)
        if header is None:
            return set()
        try:
            col = header.index("entity_id")
        except ValueError:
            col = 0
        return {r[col].strip() for r in rdr if r}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    issues: list[str] = []
    _, match, i1 = read_id_list(args.matching)
    _, cand, i2 = read_id_list(args.candidate)
    issues += i1 + i2

    s1 = read_ids(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = read_ids(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = read_ids(os.path.join(args.test_dir, "test_source3.tsv"))
    valid = s2 | s3

    missing = s1 - set(match)
    if missing:
        issues.append(f"matching: {len(missing)} test source-1 entities have no row "
                      f"(e.g. {sorted(missing)[:5]}) — this causes rejection")
    extra = set(match) - s1
    if extra:
        issues.append(f"matching: {len(extra)} rows are not test source-1 entities "
                      f"(e.g. {sorted(extra)[:5]})")

    for name, table in (("matching", match), ("candidate", cand)):
        bad_self, bad_unknown = set(), set()
        for sid, lst in table.items():
            for x in lst:
                if x in s1:
                    bad_self.add(x)
                elif x not in valid:
                    bad_unknown.add(x)
        if bad_self:
            issues.append(f"{name}: {len(bad_self)} source-1 ids used as matches "
                          f"(e.g. {sorted(bad_self)[:5]})")
        if bad_unknown:
            issues.append(f"{name}: {len(bad_unknown)} ids not present in the test set "
                          f"(e.g. {sorted(bad_unknown)[:5]})")

    not_subset = 0
    for sid, lst in match.items():
        cs = set(cand.get(sid, []))
        if not set(lst).issubset(cs):
            not_subset += 1
    if not_subset:
        issues.append(f"{not_subset} entities have matched ids that never appeared as "
                      "candidates — your candidate_pairs.tsv does not cover your matches")

    sizes = collections.Counter(len(v) for v in match.values())
    n = sum(sizes.values()) or 1
    print("predicted set sizes:")
    for k in sorted(sizes):
        print(f"  {k:>3}: {sizes[k]:>8}  ({sizes[k]/n:6.2%})")
    print(f"predicted singletons: {sizes.get(0,0)/n:.2%}")
    print(f"mean candidates per entity: {sum(len(v) for v in cand.values())/max(1,len(cand)):.1f}")

    if issues:
        print("\nFAIL")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
