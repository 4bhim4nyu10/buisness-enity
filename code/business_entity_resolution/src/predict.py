"""Produce the two submission files.

    python -m src.predict --data-dir dataset --artifacts artifacts --out output
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

from .decide import DecisionConfig
from .model import TwoStageMatcher
from .pipeline import _log, predict_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="output")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-candidates", type=int, default=60)
    ap.add_argument("--no-unique", action="store_true")
    args = ap.parse_args()

    model = TwoStageMatcher.load(os.path.join(args.artifacts, "model.pkl"))
    with open(os.path.join(args.artifacts, "corpus_stats.pkl"), "rb") as fh:
        stats = pickle.load(fh)

    cfg_path = os.path.join(args.artifacts, "decision.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            spec = json.load(fh)
        _log(f"loaded decision rule '{spec.get('strategy')}' from {cfg_path}")
    else:
        from dataclasses import asdict
        spec = {"strategy": "expected_f", **asdict(DecisionConfig())}
        _log("no decision.json found — falling back to default expected-F selection")

    predict_pipeline(
        data_dir=args.data_dir,
        model=model,
        stats=stats,
        out_dir=args.out,
        decision=spec,
        max_candidates=args.max_candidates,
        enforce_unique=not args.no_unique,
        split=args.split,
    )
    _log(
        "now run:  python3 utils/validate_submission.py "
        f"--matching {args.out}/matching_results.tsv "
        f"--candidate {args.out}/candidate_pairs.tsv --test-dir {args.data_dir}/test"
    )


if __name__ == "__main__":
    main()
