"""Generate a small synthetic dataset in the challenge's exact file format.

Purpose: smoke-test the whole pipeline (and your own changes to it) in seconds,
without touching the real data. The noise model deliberately mimics the noise
patterns the problem statement lists — abbreviation swaps, legal-suffix drift,
transliteration, token reordering, address component loss, landmark references.

    python tests/make_synthetic.py --out /tmp/synth --n 1200
"""

from __future__ import annotations

import argparse
import os
import random

FIRST = ["shree", "sri", "royal", "golden", "blue", "united", "apex", "sunrise", "global",
         "prime", "metro", "coastal", "bright", "grand", "silver", "noble", "orient"]
MID = ["kumar", "sharma", "patel", "iyer", "reddy", "singh", "smith", "johnson", "miller",
       "dupont", "laurent", "moreau", "krishna", "lakshmi", "venkat", "anderson"]
TAIL = ["traders", "enterprises", "industries", "technologies", "solutions", "foods",
        "textiles", "logistics", "motors", "pharma", "exports", "systems", "services"]
SUFFIX = {"US": ["inc", "llc", "corp", "co"], "India": ["pvt ltd", "private limited", "ltd"],
          "France": ["sarl", "sas", "sa", "eurl"]}
STREETS = {"US": ["main street", "oak avenue", "park road", "5th avenue", "elm drive"],
           "India": ["mg road", "gandhi nagar", "nehru street", "anna salai", "brigade road"],
           "France": ["rue de la paix", "avenue victor hugo", "boulevard saint germain"]}
CITY = {"US": ["austin", "denver", "boston", "seattle"],
        "India": ["chennai", "pune", "jaipur", "kochi"],
        "France": ["lyon", "nantes", "toulouse", "lille"]}

ABBR = [("street", "st"), ("road", "rd"), ("avenue", "ave"), ("private limited", "pvt ltd"),
        ("corporation", "corp"), ("incorporated", "inc"), ("limited", "ltd"), ("and", "&")]
TRANSLIT = [("sh", "s"), ("ee", "i"), ("aa", "a"), ("ksh", "x"), ("oo", "u"), ("v", "w")]


def noisy_name(name: str, rng: random.Random) -> str:
    s = name
    if rng.random() < 0.45:
        for a, b in ABBR:
            if a in s and rng.random() < 0.6:
                s = s.replace(a, b)
                break
    if rng.random() < 0.25:
        for a, b in TRANSLIT:
            if a in s and rng.random() < 0.5:
                s = s.replace(a, b)
                break
    if rng.random() < 0.2:
        toks = s.split()
        if len(toks) > 2:
            i = rng.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
            s = " ".join(toks)
    if rng.random() < 0.18 and len(s) > 6:  # typo
        i = rng.randrange(len(s))
        s = s[:i] + s[i + 1:]
    if rng.random() < 0.15:
        s = s.upper()
    if rng.random() < 0.12:
        s = s.replace(" ", ", ", 1)
    return s


def noisy_addr(addr: str, country: str, rng: random.Random) -> str:
    parts = [p.strip() for p in addr.split(",")]
    if rng.random() < 0.3 and len(parts) > 2:
        parts.pop(rng.randrange(len(parts)))
    if rng.random() < 0.2:
        parts.insert(0, f"near {rng.choice(['sbi atm', 'city mall', 'the station', 'post office'])}")
    s = ", ".join(parts)
    if rng.random() < 0.4:
        for a, b in ABBR:
            if a in s:
                s = s.replace(a, b)
                break
    if rng.random() < 0.1:
        s = ""
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/synth")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hard", action="store_true",
                    help="near-twin distractors that share most of the name and the whole "
                         "address — this is the regime where the decision layer earns its keep")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    hard = args.hard

    for split in ("train", "test"):
        os.makedirs(os.path.join(args.out, split), exist_ok=True)

    def build(split: str, n: int, countries: list[str], start: int):
        s1, s2, s3, gt = [], [], [], []
        c2 = c3 = start
        for i in range(n):
            country = rng.choice(countries)
            base = f"{rng.choice(FIRST)} {rng.choice(MID)} {rng.choice(TAIL)}"
            full = f"{base} {rng.choice(SUFFIX[country])}"
            num = rng.randrange(1, 400)
            addr = f"{num} {rng.choice(STREETS[country])}, {rng.choice(CITY[country])}, {rng.randrange(10000, 99999)}"
            sid = f"S1-{start + i:06d}"
            s1.append((sid, full, addr, country))

            matches = []
            for src, bucket in ((2, s2), (3, s3)):
                k = rng.choices([0, 1, 2], weights=[0.45, 0.45, 0.10])[0]
                for _ in range(k):
                    if src == 2:
                        c2 += 1
                        tid = f"S2-{c2:06d}"
                    else:
                        c3 += 1
                        tid = f"S3-{c3:06d}"
                    bucket.append((tid, noisy_name(full, rng), noisy_addr(addr, country, rng), country))
                    matches.append(tid)
            gt.append((sid, ",".join(matches)))

            # distractors: same neighbourhood, different business
            if rng.random() < 0.7:
                c2 += 1
                other = f"{rng.choice(FIRST)} {rng.choice(MID)} {rng.choice(TAIL)} {rng.choice(SUFFIX[country])}"
                s2.append((f"S2-{c2:06d}", other, addr, country))

            if hard:
                # near twins: same first two name tokens and the SAME address,
                # differing only in the trade word. Genuinely ambiguous — the
                # model should return probabilities in the middle of the range,
                # which is exactly where set selection beats a threshold.
                for src in (2, 3):
                    if rng.random() > 0.55:
                        continue
                    stem = " ".join(base.split()[:2])
                    twin = f"{stem} {rng.choice(TAIL)} {rng.choice(SUFFIX[country])}"
                    twin_addr = addr if rng.random() < 0.6 else noisy_addr(addr, country, rng)
                    if src == 2:
                        c2 += 1
                        s2.append((f"S2-{c2:06d}", noisy_name(twin, rng), twin_addr, country))
                    else:
                        c3 += 1
                        s3.append((f"S3-{c3:06d}", noisy_name(twin, rng), twin_addr, country))
        return s1, s2, s3, gt

    tr1, tr2, tr3, gt = build("train", args.n, ["US", "India"], 1)
    te1, te2, te3, _ = build("test", max(200, args.n // 3), ["US", "India", "France"], 900000)

    def write(path, rows, header):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for r in rows:
                fh.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")

    H = ["entity_id", "business_name", "business_address", "country"]
    write(f"{args.out}/train/train_source1.tsv", tr1, H)
    write(f"{args.out}/train/train_source2.tsv", tr2, H)
    write(f"{args.out}/train/train_source3.tsv", tr3, H)
    write(f"{args.out}/train/train_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])
    write(f"{args.out}/test/test_source1.tsv", te1, H)
    write(f"{args.out}/test/test_source2.tsv", te2, H)
    write(f"{args.out}/test/test_source3.tsv", te3, H)
    print(f"wrote synthetic dataset to {args.out}: "
          f"train s1={len(tr1)} s2={len(tr2)} s3={len(tr3)}, test s1={len(te1)}")


if __name__ == "__main__":
    main()
