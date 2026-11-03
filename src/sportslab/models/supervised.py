"""Logistic regression and XGBoost on pre-game features, judged against the Elo baseline.

Protocol (all temporal, never shuffled):

* Stage A, model selection: fit on ``train`` seasons, score on ``validation`` seasons.
  XGBoost's number of trees is chosen by early stopping on the *last training season*, so the
  validation seasons are never used to pick it.
* Stage B, final test: refit on train + validation (same settings, fixed number of trees) and
  score the ``test`` season once.

Neutral-site games are excluded from fitting and scoring (their "home" label is arbitrary).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sportslab.models.metrics import paired_log_loss_gain, score_all

ELO_ONLY = ["elo_diff"]
ALL_FEATURES = ["elo_diff", "form_diff", "rest_diff", "home_b2b", "away_b2b"]

_XGB_DEFAULTS: dict[str, Any] = {
    "max_depth": 2,
    "learning_rate": 0.03,
    "min_child_weight": 30,
    "subsample": 0.8,
    "reg_lambda": 5.0,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 1,
}
_MAX_TREES = 1000
_EARLY_STOPPING = 50


def make_logistic() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def make_xgb(n_estimators: int, early_stopping_rounds: int | None = None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **_XGB_DEFAULTS
    )


@dataclass(frozen=True)
class SupervisedExperiment:
    scores_validation: pd.DataFrame  # models x metrics, fit on train
    scores_test: pd.DataFrame  # models x metrics, fit on train + validation
    gain_vs_elo_validation: pd.DataFrame  # per-game log-loss gain over Elo, with 95% interval
    gain_vs_elo_test: pd.DataFrame
    test_predictions: pd.DataFrame
    best_iteration: int
    logistic_coefficients: pd.Series  # standardized features, final model
    xgb_importance: pd.Series  # gain-based, final model


def _season_mask(df: pd.DataFrame, seasons: Sequence[str]) -> pd.Series:
    return df["season"].isin(list(seasons))


def _gains(y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {
        name: dict(
            zip(
                ("gain", "ci_low", "ci_high"), paired_log_loss_gain(y, p, preds["elo"]), strict=True
            )
        )
        for name, p in preds.items()
        if name != "elo"
    }
    return pd.DataFrame(rows).T


def _predict(
    fit: pd.DataFrame, target: pd.DataFrame, models: dict[str, Callable[[], Any]]
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {
        "home rate": np.full(len(target), fit["home_win"].mean()),
        "elo": target["p_elo"].to_numpy(),
    }
    for name, (cols, factory) in models.items():
        model = factory()
        model.fit(fit[cols], fit["home_win"])
        preds[name] = model.predict_proba(target[cols])[:, 1]
    return preds


def run_supervised_experiment(
    features: pd.DataFrame,
    train: Sequence[str],
    validation: Sequence[str],
    test: str,
) -> SupervisedExperiment:
    data = features[~features["neutral_site"]]
    train, validation = list(train), list(validation)
    tr = data[_season_mask(data, train)]
    va = data[_season_mask(data, validation)]
    te = data[data["season"] == test]

    # How many trees? Early stopping on the last *training* season, never on validation/test.
    stop_season = sorted(train)[-1]
    fit_part, stop_part = tr[tr["season"] != stop_season], tr[tr["season"] == stop_season]
    stopper = make_xgb(_MAX_TREES, early_stopping_rounds=_EARLY_STOPPING)
    stopper.fit(
        fit_part[ALL_FEATURES],
        fit_part["home_win"],
        eval_set=[(stop_part[ALL_FEATURES], stop_part["home_win"])],
        verbose=False,
    )
    best_iteration = int(stopper.best_iteration)
    n_trees = best_iteration + 1

    models = {
        "logistic (elo only)": (ELO_ONLY, make_logistic),
        "logistic (all)": (ALL_FEATURES, make_logistic),
        "xgboost (all)": (ALL_FEATURES, lambda: make_xgb(n_trees)),
    }
    val_preds = _predict(tr, va, models)
    test_preds = _predict(pd.concat([tr, va]), te, models)

    final_fit = pd.concat([tr, va])
    final_logit = make_logistic().fit(final_fit[ALL_FEATURES], final_fit["home_win"])
    final_xgb = make_xgb(n_trees).fit(final_fit[ALL_FEATURES], final_fit["home_win"])

    def to_scores(y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame({name: score_all(y, p) for name, p in preds.items()}).T

    y_val, y_test = va["home_win"].to_numpy(), te["home_win"].to_numpy()
    ids = [
        c
        for c in ("game_id", "season", "game_date", "home_team", "away_team", "home_win")
        if c in te
    ]
    test_predictions = pd.concat([te[ids].reset_index(drop=True), pd.DataFrame(test_preds)], axis=1)
    return SupervisedExperiment(
        scores_validation=to_scores(y_val, val_preds),
        scores_test=to_scores(y_test, test_preds),
        gain_vs_elo_validation=_gains(y_val, val_preds),
        gain_vs_elo_test=_gains(y_test, test_preds),
        test_predictions=test_predictions,
        best_iteration=best_iteration,
        logistic_coefficients=pd.Series(
            final_logit[-1].coef_[0], index=ALL_FEATURES, name="standardized coefficient"
        ),
        xgb_importance=pd.Series(final_xgb.feature_importances_, index=ALL_FEATURES, name="gain"),
    )
