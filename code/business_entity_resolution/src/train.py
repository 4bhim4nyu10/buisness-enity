"""Train the matcher, select a decision rule, report the honest score.

    python -m src.train --data-dir dataset --out artifacts
    python -m src.train --data-dir dataset --out artifacts --holdout-country India

The second form is your France simulator: every Source-1 entity from that
country is excluded from all training folds and from the calibrator, then
scored with a decision rule chosen on the *other* countries only. If that
number is far below the overall out-of-fold score, your pipeline is
memorising country-specific surface patterns and France will hurt.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

from .dataio import load_ground_truth, load_sources
from .evaluate import score_breakdown
from .pipeline import _log, leave_one_country_out, scores_by_entity, train_pipeline, validate_oof
from .tune import ablation_table, print_ablation, select_strategy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--max-candidates", type=int, default=60)
    ap.add_argument("--no-unique", action="store_true",
                    help="disable the unique-target constraint (see src.analyze output)")
    ap.add_argument("--holdout-country", default=None,
                    help="exclude this country from training entirely and score it separately")
    ap.add_argument("--loco", action="store_true",
                    help="select the decision rule on pooled leave-one-country-out predictions "
                         "(transfer conditions, like France). Costs one model fit per country.")
    ap.add_argument("--sample-size", type=int, default=15000,
                    help="subsample size for training (0 for full set)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    enforce_unique = not args.no_unique

    try:
        test_ss = load_sources(args.data_dir, "test", nrows=100_000 if args.sample_size else None)
    except FileNotFoundError:
        test_ss = None
        _log("no test split found — corpus stats will be fitted on train only")

    model, stats, prep, pairs, X, y = train_pipeline(
        args.data_dir,
        os.path.join(args.out, "model.pkl"),
        max_candidates=args.max_candidates,
        test_set=test_ss,
        holdout_country=args.holdout_country,
        sample_size=args.sample_size if args.sample_size > 0 else None,
    )
    with open(os.path.join(args.out, "corpus_stats.pkl"), "wb") as fh:
        pickle.dump(stats, fh)

    truth = load_ground_truth(args.data_dir, "train")
    all_s1 = list(prep.s1.keys())
    truth = {k: v for k, v in truth.items() if k in prep.s1}
    holdout = getattr(model, "holdout_entities_", set())

    sbe_all = scores_by_entity(pairs, model.oof_calibrated_, all_s1)
    # Select the rule on entities the model was actually trained around.
    sbe_fit = {k: v for k, v in sbe_all.items() if k not in holdout}
    truth_fit = {k: v for k, v in truth.items() if k not in holdout}

    if args.loco:
        _log("leave-one-country-out: refitting once per country")
        sbe_sel = leave_one_country_out(X, y, pairs, prep)
        truth_sel = {k: v for k, v in truth.items() if k in sbe_sel}
        _log(f"selecting the decision rule on {len(truth_sel)} out-of-country entities")
    else:
        sbe_sel, truth_sel = sbe_fit, truth_fit
        _log("selecting the decision rule on out-of-fold predictions")

    spec, best, _trace = select_strategy(sbe_sel, truth_sel, enforce_unique=enforce_unique)
    _log(f"selected rule -> macro F0.5 on the selection set = {best:.4f}")
    print(json.dumps(spec, indent=2))

    with open(os.path.join(args.out, "decision.json"), "w") as fh:
        json.dump(spec, fh, indent=2)

    bd, _, preds = validate_oof(model, pairs, all_s1, truth_fit, spec, enforce_unique)
    print("\n--- OUT-OF-FOLD SCORE (trained countries) -----------------------")
    print(bd)

    print("\n--- DECISION-RULE ABLATION --------------------------------------")
    print_ablation(ablation_table(sbe_fit, truth_fit, spec))

    if holdout:
        truth_ho = {k: v for k, v in truth.items() if k in holdout}
        print(f"\n--- HELD-OUT COUNTRY: {args.holdout_country} "
              "(never seen in training) ---")
        print(score_breakdown({k: preds.get(k, []) for k in truth_ho}, truth_ho))
        print("  Compare this against the trained-country score above. A gap of more\n"
              "  than a couple of points is your expected France penalty.")

    imp = model.feature_importance()
    if imp:
        print("\n--- TOP 25 FEATURES ---------------------------------------------")
        for n, v in imp[:25]:
            print(f"  {n:<28} {v:.0f}")

    _log("done. run `python -m src.predict` to produce the submission files.")


if __name__ == "__main__":
    main()
