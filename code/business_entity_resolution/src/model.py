"""
The matching model: a two-stage gradient-boosted classifier with isotonic
calibration.

Two stages, because the relational features ("is this the best candidate for
this entity?", "is this Source-2 record better explained by a different
Source-1 entity?") need a score to rank by:

    stage 1 : intrinsic features            -> out-of-fold score s1
    stage 2 : intrinsic + relational(s1)    -> final score

The out-of-fold construction is the important detail. If stage 2 sees
relational features derived from in-fold stage-1 predictions, those features are
sharper at training time than they will ever be at inference time, and the model
learns to trust them too much. Building them from OOF predictions keeps train
and inference distributions identical.

Calibration matters more here than in a typical competition: the decision layer
consumes probabilities as probabilities, not as a ranking. Isotonic regression
on out-of-fold predictions is cheap and monotone, so it cannot reorder anything
— it only fixes the scale.

LightGBM (MIT) is used when available; scikit-learn's HistGradientBoosting is a
drop-in fallback so the pipeline never hard-fails on environment.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

try:  # pragma: no cover
    import lightgbm as lgb

    _HAVE_LGB = True
except Exception:  # pragma: no cover
    _HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from .features import (
    BASE_FEATURES,
    CHANNEL_FEATURES,
    FEATURE_NAMES,
    Pair,
    add_relational,
)

N_BASE = len(BASE_FEATURES) + len(CHANNEL_FEATURES)


DEFAULT_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=700,
    verbose=-1,
)


def _new_model(params: dict, seed: int):
    if _HAVE_LGB:
        p = dict(params)
        p["seed"] = seed
        return lgb.LGBMClassifier(**p)
    return HistGradientBoostingClassifier(
        max_iter=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_leaf_nodes=params.get("num_leaves", 63),
        min_samples_leaf=params.get("min_data_in_leaf", 50),
        l2_regularization=params.get("lambda_l2", 1.0),
        random_state=seed,
    )


def _predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


@dataclass
class TwoStageMatcher:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    n_folds: int = 5
    seed: int = 13
    stage1: list = field(default_factory=list)
    stage2: list = field(default_factory=list)
    calibrator: IsotonicRegression | None = None
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    # ------------------------------------------------------------------ #

    def _entity_folds(
        self, pairs: Sequence[Pair], holdout: set[str] | None = None
    ) -> np.ndarray:
        """Fold id per row, assigned by Source-1 entity so no entity straddles a
        fold boundary (its relational features would leak across).

        Entities in `holdout` get fold -1: never trained on, always predicted by
        the full ensemble. That is how the country holdout stays honest — a
        'holdout' the model trained on tells you nothing about France.
        """
        rng = np.random.default_rng(self.seed)
        holdout = holdout or set()
        ents = sorted({p.s1_id for p in pairs})
        assign = {e: (-1 if e in holdout else int(rng.integers(self.n_folds))) for e in ents}
        return np.asarray([assign[p.s1_id] for p in pairs], dtype=np.int16)

    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_base: np.ndarray,
        y: np.ndarray,
        pairs: Sequence[Pair],
        holdout_entities: set[str] | None = None,
    ) -> "TwoStageMatcher":
        folds = self._entity_folds(pairs, holdout_entities)
        self.holdout_mask_ = folds == -1
        if self.holdout_mask_.any():
            n_ho = len({p.s1_id for p, h in zip(pairs, self.holdout_mask_) if h})
            print(f"  holding out {n_ho} source-1 entities from all training folds")
        n = len(y)

        # ---- stage 1 : OOF intrinsic score -----------------------------
        oof1 = np.zeros(n, dtype=np.float64)
        self.stage1 = []
        for f in range(self.n_folds):
            tr, va = (folds != f) & (folds != -1), folds == f
            if tr.sum() == 0 or va.sum() == 0:
                continue
            m = _new_model(self.params, self.seed + f)
            m.fit(X_base[tr], y[tr])
            oof1[va] = _predict(m, X_base[va])
            self.stage1.append(m)
        if self.stage1 and self.holdout_mask_.any():
            oof1[self.holdout_mask_] = np.mean(
                [_predict(m, X_base[self.holdout_mask_]) for m in self.stage1], axis=0
            )
        if not self.stage1:  # degenerate tiny dataset
            m = _new_model(self.params, self.seed)
            m.fit(X_base, y)
            self.stage1 = [m]
            oof1 = _predict(m, X_base)

        # ---- stage 2 : intrinsic + relational(OOF stage 1) --------------
        X_full = add_relational(X_base, pairs, oof1)
        oof2 = np.zeros(n, dtype=np.float64)
        self.stage2 = []
        for f in range(self.n_folds):
            tr, va = (folds != f) & (folds != -1), folds == f
            if tr.sum() == 0 or va.sum() == 0:
                continue
            m = _new_model(self.params, 100 + self.seed + f)
            m.fit(X_full[tr], y[tr])
            oof2[va] = _predict(m, X_full[va])
            self.stage2.append(m)
        if not self.stage2:
            m = _new_model(self.params, 100 + self.seed)
            m.fit(X_full, y)
            self.stage2 = [m]
            oof2 = _predict(m, X_full)
        elif self.holdout_mask_.any():
            oof2[self.holdout_mask_] = np.mean(
                [_predict(m, X_full[self.holdout_mask_]) for m in self.stage2], axis=0
            )

        # ---- calibration -------------------------------------------------
        # Fit the calibrator on non-holdout rows only, so the holdout score is
        # not flattered by a calibrator that has seen its labels.
        fit_mask = ~self.holdout_mask_
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(oof2[fit_mask], y[fit_mask])

        self.oof_raw_ = oof2
        self.oof_calibrated_ = self.calibrator.predict(oof2)
        self.oof_stage1_ = oof1
        return self

    # ------------------------------------------------------------------ #

    def predict_proba(self, X_base: np.ndarray, pairs: Sequence[Pair]) -> np.ndarray:
        if X_base.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        s1 = np.mean([_predict(m, X_base) for m in self.stage1], axis=0)
        X_full = add_relational(X_base, pairs, s1)
        s2 = np.mean([_predict(m, X_full) for m in self.stage2], axis=0)
        return self.calibrator.predict(s2) if self.calibrator is not None else s2

    # ------------------------------------------------------------------ #

    def feature_importance(self) -> list[tuple[str, float]]:
        if not _HAVE_LGB or not self.stage2:
            return []
        imp = np.mean([m.feature_importances_ for m in self.stage2], axis=0)
        names = self.feature_names[: len(imp)]
        return sorted(zip(names, imp.tolist()), key=lambda kv: -kv[1])

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "TwoStageMatcher":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Label construction
# --------------------------------------------------------------------------- #


def make_labels(pairs: Sequence[Pair], truth: dict[str, set[str]]) -> np.ndarray:
    """1 when the candidate pair is in the ground truth.

    Note that negatives come from the *same blocker* used at inference. Training
    on random negatives would produce a model that is excellent at rejecting
    obviously unrelated businesses and useless at the near-duplicates that
    actually decide the score.
    """
    return np.asarray(
        [1 if p.t_id in truth.get(p.s1_id, ()) else 0 for p in pairs], dtype=np.int8
    )
