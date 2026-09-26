# Business Entity Resolution — Amazon ML Challenge 2026

A blocking + classifier + F_0.5-aware decision-layer pipeline for matching business
records across three noisy, independently-sourced data feeds. Built for the Amazon ML
Challenge "Business Entity Resolution" hackathon.

**Status:** pipeline is complete and verified end-to-end on the real training/test data
(see `docs/` for logs and numbers). It has **not** yet been run at full dataset scale —
see [Scale & next steps](#scale--next-steps) below before you submit to the leaderboard.

## What's here

```
.
├── code/business_entity_resolution/   # the pipeline — see its own README for CLI details
│   ├── src/                           # normalize → block → featurize → score → decide
│   ├── tests/                         # unit tests (decision-layer DP verified vs brute force)
│   ├── utils/validate_submission.py   # official-format local validator (stdlib only)
│   └── requirements.txt
├── dataset/
│   ├── train/                          # 8k-entity real-data sample (true matches + distractors)
│   └── test/                           # 3k-entity real-data sample, includes France
├── artifacts/
│   ├── trained/                       # model + calibrator + decision rule, fit on sample_train
│   └── loco_holdout/                  # same, but with India held out entirely (France proxy)
├── output/                            # matching_results.tsv + candidate_pairs.tsv
│                                       # (from sample_test — NOT the full leaderboard submission, see below)
└── docs/
    ├── Documentation_template.md      # filled-in methodology write-up, official template
    └── *_output.txt                   # raw logs from every real run below, for auditability
```

## Real results (on an 8k-entity sample of the actual training data)

| Metric | Score |
|---|---|
| Blocking entity recall | 98.3% |
| Blocking pair recall | 99.4% |
| Out-of-fold macro F_0.5 (trained countries) | **0.9838** |
| Out-of-fold macro F_0.5, refit with `--loco` | 0.9918 |
| Held-out country (India excluded from training — France proxy) | **0.9630** |
| → generalization gap | **2.9 points** |

Full breakdowns, ablations, and the reasoning behind the decision layer are in
`docs/Documentation_template.md`.

## Quickstart (on the included sample data)

```bash
cd code/business_entity_resolution
pip install -r requirements.txt

# reproduce the numbers above
python -m src.train --data-dir ../../dataset --out ../../artifacts/trained
python -m src.train --data-dir ../../dataset --out ../../artifacts/loco_holdout --holdout-country India

# reproduce output/
python -m src.predict --data-dir ../../dataset --artifacts ../../artifacts/trained --out ../../output

# validate format
python3 utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

`dataset/train` and `dataset/test` here are the real-data samples described above — the
commands work as-is with no renaming.

## Scale & next steps

The real dataset is much larger than the sample checked in here — roughly 2.2M Source-1
entities and ~5M records each in Source 2/3 (12M+ rows, ~2GB of TSVs). That's excluded
from this repo (see `.gitignore`) both for repo size and because it wasn't feasible to
process end-to-end in the sandbox this was built in (1 CPU, ~3GB RAM — an 8k-entity
subsample took ~11 minutes; the full set would take on the order of days there).

To finish this for the actual leaderboard:

1. Put the real `train/` and `test/` folders (as given by the challenge) under `dataset/`.
2. Run, on real hardware:
   ```bash
   python -m src.analyze  --data-dir dataset
   python -m src.train    --data-dir dataset --out artifacts --loco
   python -m src.predict  --data-dir dataset --artifacts artifacts --out output
   python3 utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```
3. Confirm `PASS`, then upload `output/matching_results.tsv` to the leaderboard.
4. Update the bracketed/sample-scale numbers in `docs/Documentation_template.md` with the
   full-scale results before zipping the final submission package (structure described
   in `docs/Documentation_template.md`'s Appendix A).

`--loco` is recommended for the real run given the confirmed ~2.9-point country
generalization gap above — it tunes the decision rule against pooled
leave-one-country-out predictions instead of plain in-country ones, which better matches
the France condition in the real test set.

## License

Pipeline code: MIT (see `LICENSE`). Model layer uses LightGBM (MIT) when available,
falling back to scikit-learn (BSD) — both comply with the challenge's MIT/Apache-2.0,
≤8B-parameter model constraint.
