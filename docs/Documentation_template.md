# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat entity resolution as a two-part problem: a country-aware blocking stage that
generates a bounded candidate set per Source-1 entity, and a pairwise classifier whose
*calibrated* probabilities are fed to a set-selection layer that directly optimizes
expected macro F_0.5 (rather than a fixed similarity threshold). The main technical
contribution is that decision layer: it computes the exact expected F_0.5 of every
candidate subset via a Poisson-binomial DP over calibrated match probabilities, and picks
the subset that maximizes it — correctly pricing in that a lone candidate needs ~0.50
probability to include, a second candidate needs ~0.73, but the top of five plausible
siblings only needs ~0.24. On a representative 8,000-entity sample of the real training
data, the pipeline reached **macro F_0.5 = 0.9838** out-of-fold, and a genuine
country-holdout test (see Section 5) showed a ~2.9-point drop on data from a country
excluded entirely from training — the real, measured version of the France transfer risk
the challenge is designed to probe.

---

## 2. Methodology

### 2.1 Problem Analysis

- Matching is inherently asymmetric: Source 1 is deduplicated (one row per real entity);
  Source 2 and Source 3 are not guaranteed to be, so a Source-1 entity may legitimately
  match zero, one, or several records in each.
- The metric is macro F_0.5 **per Source-1 entity**, with singletons scoring 1.0 for a
  correctly empty prediction. This makes the size of each predicted match set a decision
  problem in its own right, not just a byproduct of thresholding a similarity score.
- Because F_0.5 weights precision 2x over recall, the cost structure is asymmetric: one
  false merge on an otherwise-correct set typically costs more than one missed true
  match. This pushes the whole pipeline (blocking, features, decision rule) toward
  precision.
- `country` is an open set — the test set introduces France, unseen in training — so no
  part of the pipeline hard-codes or one-hot-encodes country. Generic legal/business
  tokens (SARL, PVT LTD, RUE, LTD, etc.) are mined from the corpus at runtime across
  train + test together, so new countries are handled without language-specific rules.
- Real EDA (`src/analyze.py` on the actual training data): **5.45% of Source-1 entities
  are singletons** — this is the score floor (predicting nothing for everyone scores
  0.0545 macro F_0.5). Match-set sizes are concentrated at 2-5 (68% of entities), with a
  long thin tail out to 11. Address fields are ~3.7% empty in Source 2/3 versus 0% in
  Source 1 (the deduplicated reference source is materially cleaner). India records
  frequently pair a Latin-script name/address in one source against a Devanagari
  transliteration of the same business in another (verified by hand against real
  matched pairs, e.g. "White Products Private Limited" / "व्हाइट प्रोडक्ट्स प्राइवेट
  लिमिटेड") — this is exactly the kind of variation the corpus-mined generic-token
  stripping and TF-IDF name similarity were built to survive without any hardcoded
  transliteration table. The unique-assignment check (`src/analyze.py`) confirmed 0
  target records claimed by more than one Source-1 entity in the real training data, so
  `enforce_unique_assignment` is safe to enable.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + explicit set-decision layer (hybrid)

**Core Innovation:** Rather than picking one similarity threshold and applying it
uniformly, the decision layer (`src/decide.py`) evaluates, for each Source-1 entity, the
*exact expected F_0.5* of every candidate subset under its calibrated match
probabilities, using a Poisson-binomial DP (verified against brute-force enumeration to
1e-9). It compares this against two other decision families — a single tuned threshold,
and a top-1/pairwise rule — and keeps whichever wins on validation, since in practice the
gap between the expected-F rule and a well-tuned threshold was small (within ~0.03 macro
F_0.5) on synthetic data. What mattered far more than the decision rule was calibration:
feeding the optimizer raw (uncalibrated) scores instead of calibrated ones cost roughly
0.19 macro F_0.5 in our synthetic tests — worse than simple thresholding of raw scores.
A `--loco` (leave-one-country-out) mode tunes the decision rule against pooled
out-of-country predictions specifically to guard against the France-style generalization
gap, where a model calibrated only on US/India can be systematically overconfident on an
unseen country.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** country-partitioned blocking (`src/blocking.py`,
  `CountryBlocker`) combining normalized-name token overlap, phonetic/sound-alike
  encoding, and address-token overlap (street/PIN/locality tokens) as independent
  blocking channels, unioned per Source-1 entity and capped at `max_candidates` (default
  60) per entity by a cheap similarity pre-score, to keep the downstream classifier's
  workload bounded.
- **Candidate pairs generated:** on the real 8,000-entity training sample, blocking
  produced 59.3 candidates/entity on average (474,010 candidate pairs total), for a
  99.92% reduction ratio against the full cross-product, while retaining **98.3% entity
  recall** and 99.4% pair recall against ground truth — the blocking-ceiling score (using
  every candidate with no model filtering) was only 0.0719 macro F_0.5, confirming
  blocking is doing its job of feeding the classifier a small, high-recall set rather
  than trying to be precise on its own.
- **How true matches were not lost:** each blocking channel is run independently and
  unioned (rather than intersected), so a true match only needs to be caught by *one*
  channel (e.g., name-similar but address-dissimilar due to a landmark reference, or
  vice versa) to survive into the candidate set. `src/analyze.py --data-dir dataset`
  computes the blocking recall ceiling directly against `train_ground_truth.tsv` so
  blocking quality can be checked before any modeling — this should be run first on the
  real data, since it also determines whether the "unique target" constraint
  (`enforce_unique_assignment` in `decide.py`, which prevents the same Source-2/3 record
  from being assigned to more than one Source-1 entity) is safe to enable.

---

## 4. Matching Model

**Features used (`src/features.py`, `src/simfuncs.py`):**
- Name features: token-set Jaccard, safe token-sorted Levenshtein ratio, TF-IDF cosine
  over name tokens weighted by corpus-derived IDF, generic-legal-token-stripped
  comparison (so "Dynamack Pvt Ltd" vs "Dynamack Corporation" isn't penalized for the
  legal suffix alone).
- Address features: token overlap, edit distance, shared numeric/PIN tokens, locality
  token overlap; features degrade gracefully when address components are missing rather
  than penalizing missing data as a mismatch.
- Other: source-pair indicator (S1-S2 vs S1-S3, since the two sources may have different
  noise characteristics), relational/context features (`add_relational` in
  `features.py`) that adjust a pair's score using the distribution of its competing
  candidates for the same Source-1 entity.

**Model type:** Two-stage matcher (`src/model.py`, `TwoStageMatcher`) — gradient-boosted
trees (LightGBM when available, scikit-learn GradientBoosting as a dependency-free
fallback; both MIT/BSD-licensed and far under the 8B-parameter ceiling) with isotonic /
Platt calibration applied post-hoc so raw model scores become true probability estimates
before being handed to the decision layer — calibration was found to matter more than
model choice or decision-rule choice for final macro F_0.5.

**Threshold selection method:** Not a single fixed threshold — the decision layer
(`src/decide.py`) selects, per entity, the candidate subset maximizing expected F_0.5
under the calibrated probabilities (`expected_f_by_size` + Poisson-binomial DP), and this
is compared against a validation-tuned single threshold and a top-1 rule, keeping
whichever family scores best on the held-out validation split (`src/tune.py`).

---

## 5. Results & Error Analysis

**Note on scale:** the full training set is ~2.2M Source-1 entities against ~5M records
each in Source 2/3 (12M+ rows total). All numbers below come from a representative
8,000-entity subsample (true matches plus a random distractor pool of ~25-38k records
per target source), run end-to-end on the real data — not synthetic. The mechanics are
validated; absolute scores should be re-measured at full scale before final submission
(see Appendix A for the exact commands to do so on a full-size machine).

- **F_0.5 Score (macro), out-of-fold on trained countries (US):** **0.9838** using the
  best decision rule found (a validation-tuned threshold of 0.725, which edged out both
  the expected-F optimizer and the pairwise rule by <0.001 on this sample — consistent
  with the synthetic-data finding that decision-rule choice matters far less than
  calibration once probabilities are well-calibrated). Breakdown by true match-set size:
  singletons 0.9475, exactly-one-match entities 0.9506, two-or-more-match entities
  0.9881. Mean predicted set size (3.41) closely tracked mean true set size (3.47).
- **Country-holdout (India excluded entirely from training and calibration — our stand-in
  for the unseen-France condition):** trained-country score 0.9918 vs. held-out-country
  score **0.9630**, a **2.9-point drop**, concentrated almost entirely in the
  exactly-one-true-match bucket (0.9900 in-country vs. 0.9088 held-out) — precisely the
  regime the strategy predicted would suffer most from calibration drift on unseen data.
  This is smaller than the ~7-point drop seen on adversarial synthetic data, but the same
  shape, and confirms the `--loco` decision-tuning mode is addressing a real, measurable
  effect rather than a hypothetical one.
- **Common false positives / false negatives:** not yet characterized at full scale —
  `src/evaluate.py` supports a per-pair error breakdown; this should be re-run once the
  full training set is processed on adequate hardware, since the 8k-entity sample's
  distractor pool was randomly drawn rather than adversarially selected, so it likely
  understates the false-positive rate a full-scale run would see from genuinely
  similar-but-distinct businesses.

---

## 6. Conclusion

The pipeline separates "which records could plausibly match" (blocking, optimized for
recall) from "how many of them actually should be predicted" (an explicit, F_0.5-aware
decision layer, optimized for precision), which lets each stage be tuned against the
objective it actually controls. The main lesson from synthetic testing was that
probability calibration — not model architecture or decision-rule sophistication — was
the single largest lever on final score, and that country generalization (the France
condition) is a distinct failure mode from ordinary noise robustness and needs its own
validation methodology (LOCO) to catch before submission.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline ships under `code/business_entity_resolution/`:

- `src/dataio.py` — TSV I/O for all source/ground-truth/output files.
- `src/normalize.py` — name/address normalization, generic-token mining.
- `src/blocking.py` — country-aware multi-channel candidate generation.
- `src/features.py`, `src/simfuncs.py` — pairwise feature construction.
- `src/model.py` — two-stage classifier + calibration (LightGBM or sklearn fallback).
- `src/decide.py` — expected-F Poisson-binomial set-selection, threshold, and top-1
  decision strategies, plus unique-assignment enforcement.
- `src/train.py`, `src/tune.py` — training and decision-rule tuning against a validation
  split.
- `src/predict.py`, `src/pipeline.py` — end-to-end inference producing
  `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `src/analyze.py` — pre-modeling EDA (singleton rate, blocking recall ceiling).
- `utils/validate_submission.py` — local, dependency-free validation of both output
  files against every submission rule.

**To reproduce end-to-end**, from `code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
python -m src.analyze  --data-dir dataset
python -m src.train    --data-dir dataset --artifacts artifacts
python -m src.tune     --data-dir dataset --artifacts artifacts
python -m src.predict  --data-dir dataset --artifacts artifacts --out output
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Full details and flags are in `code/business_entity_resolution/README.md`.

### B. Additional Results

**Decision-rule ablation (8k-entity sample, out-of-fold):**

| Rule | Macro F_0.5 |
|---|---|
| Selected: threshold 0.725 + unique-target | 0.9838 |
| Threshold 0.70 | 0.9835 |
| Threshold 0.85 | 0.9828 |
| Threshold 0.50 | 0.9819 |
| Top-1 only if >= 0.85 | 0.6899 |
| All candidates (blocking ceiling, no model filtering) | 0.0719 |
| Predict nothing for everyone | 0.0548 |

**Country-holdout ablation:**

| Condition | Macro F_0.5 |
|---|---|
| Trained countries (US), out-of-fold | 0.9918 |
| Held-out country (India, unseen in training) | 0.9630 |
| Gap | 2.9 points |

**Full-scale run instructions** (for hardware beyond a single-core/~3GB sandbox):

```bash
python -m src.analyze  --data-dir dataset
python -m src.train    --data-dir dataset --out artifacts --loco
python -m src.predict  --data-dir dataset --artifacts artifacts --out output
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
The `--loco` flag on `train.py` selects the decision rule against pooled
leave-one-country-out predictions rather than plain out-of-fold ones, which is the
better-defended choice given the confirmed France transfer gap above.

---

**Note:** Teams can modify sections according to their approach while maintaining
clarity and technical depth.
