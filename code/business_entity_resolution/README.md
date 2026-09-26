# Business Entity Resolution — pipeline

Matches Source-2 and Source-3 business records to a deduplicated Source-1
reference, and emits `matching_results.tsv` and `candidate_pairs.tsv`.

Everything runs on CPU with no external data access of any kind. The only
inputs are the files in `dataset/`.

---

## Reproduce end to end

```bash
pip install -r requirements.txt

# 0. Look at the data before anything else. This tells you your singleton
#    rate (your score floor) and whether the unique-assignment constraint holds.
python -m src.analyze  --data-dir dataset

# 1. Train. Fits corpus stats, blocks, featurises, trains the two-stage model,
#    calibrates it, and selects a decision rule on out-of-fold predictions.
python -m src.train    --data-dir dataset --out artifacts

# 2. Predict. Writes both submission files into output/.
python -m src.predict  --data-dir dataset --artifacts artifacts --out output

# 3. Validate before you spend a submission.
python utils/check_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
# ...and then the organisers' own script, which is authoritative:
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected wall-clock on a laptop: blocking and featurisation dominate. Budget
roughly a minute per 100k candidate pairs for featurisation with `rapidfuzz`
installed, several times that without it.

### Useful variants

```bash
# France simulator: exclude a country from training entirely and score it.
python -m src.train --data-dir dataset --out artifacts --holdout-country India

# Select the decision rule under transfer conditions rather than in-country.
# Costs one model fit per country; the right choice if France is a large share
# of the test set.
python -m src.train --data-dir dataset --out artifacts --loco

# Disable the unique-target constraint (do this if src.analyze reports that
# target records are legitimately claimed by more than one Source-1 entity).
python -m src.train --data-dir dataset --out artifacts --no-unique
```

---

## Layout

```
src/
  normalize.py   unicode/abbreviation normalisation; corpus-driven mining of
                 generic tokens (this is what makes France work without
                 hardcoding French legal suffixes)
  dataio.py      every TSV read and write, so the tab rule can't be broken
  blocking.py    six recall channels over pruned inverted indexes, unioned
  simfuncs.py    string similarity primitives (rapidfuzz or pure-python)
  features.py    45 pairwise features: intrinsic + relational/competitive
  model.py       two-stage LightGBM + isotonic calibration, entity-grouped OOF
  decide.py      exact expected-F_0.5 set selection, plus threshold families
  tune.py        searches all three decision-rule families, keeps the winner
  evaluate.py    the official macro F_0.5, plus a stratified breakdown
  pipeline.py    orchestration
  analyze.py     EDA — run first
  train.py       entry point
  predict.py     entry point
tests/
  test_decide.py      DP checked against brute-force enumeration
  make_synthetic.py   fake data in the challenge's format, for smoke tests
utils/
  check_submission.py local format validator
```

## Smoke test without the real data

```bash
python tests/test_decide.py
python tests/make_synthetic.py --out /tmp/synth --n 4000 --hard
python -m src.train   --data-dir /tmp/synth --out /tmp/art
python -m src.predict --data-dir /tmp/synth --artifacts /tmp/art --out /tmp/out
```

---

## Design notes

**Blocking is the ceiling.** A true match that never becomes a candidate is
unrecoverable, so six complementary channels are unioned rather than tuned
against each other. `src.train` prints pair recall and entity recall every run;
if pair recall drops below ~0.97, widen the weakest channel before touching the
model. `candidate_pairs.tsv` is written from exactly the set the model scores,
as the rules require.

**The relational features matter as much as the string features.** Source 1 is
deduplicated, so a Source-2 record can belong to at most one Source-1 entity.
Rank within the entity's candidates, margin to the runner-up, and whether the
pair is a mutual best match encode that competition directly, and they are what
separates a near-twin from the real thing. They are built from *out-of-fold*
first-stage scores so training and inference see the same distribution.

**Calibration is not optional.** The decision rule consumes probabilities
literally. In simulation, feeding raw classifier scores to the expected-F
optimiser instead of calibrated ones cost 0.19 macro F_0.5 — worse than simply
thresholding the raw scores.

**The decision rule is chosen, not assumed.** `src/decide.py` derives the exact
expected-F_0.5-optimal prediction set per entity. It is principled and needs no
threshold transferred from validation, but measured against a well-tuned
threshold it is roughly break-even — so `src/tune.py` searches the threshold,
first/rest-threshold-pair and expected-F families and keeps whichever wins.
Report that comparison honestly in your methodology document.

**Nothing here is country-specific.** No `if country == "India"` anywhere. The
generic-token vocabulary is mined from the corpus at runtime, fitted over train
*and* test records together — transductive use of provided data, not an
external lookup. French legal suffixes and street words get demoted
automatically because they are frequent in the French part of the test corpus.

## Licensing

LightGBM (MIT), scikit-learn (BSD-3), rapidfuzz (MIT), numpy/pandas (BSD-3).
The default pipeline contains no pretrained model at all. The optional
cross-encoder re-ranker uses MIT or Apache-2.0 checkpoints under 500M
parameters, comfortably inside the 8B cap.
