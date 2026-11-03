"""Predict award selections with temporal validation: train on earlier seasons, test on the next.

Baselines first (rank by points per game, by PIE), then logistic regression, then XGBoost. The
headline metric is R-precision: among a season's R real winners, how many are in the model's
top R. Probabilities are scored with log loss and Brier; differences carry paired intervals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sportslab.features.awards import as_matrix
from sportslab.models.metrics import brier_score, log_loss, paired_log_loss_gain

ModelFactory = Callable[[], object]


def make_logistic() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def make_xgb() -> object:
    # Small, fixed trees: with ~80 candidates x a few seasons there is nothing to tune on.
    return XGBClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, eval_metric="logloss"
    )


def r_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    """Share of the top-R scored candidates that are real winners, R = number of winners."""
    r = int(truth.sum())
    if r == 0:
        return float("nan")
    top = np.argsort(-np.asarray(scores), kind="stable")[:r]
    return float(np.asarray(truth)[top].mean())


def winner_rank(scores: np.ndarray, truth: np.ndarray) -> int:
    """Rank (1 = best) of the single winner; ties count against the winner (pessimistic)."""
    winner = np.flatnonzero(truth)[0]
    return int((np.asarray(scores) >= scores[winner]).sum())


def expanding_window(
    panels: Mapping[str, pd.DataFrame],
    features: list[str],
    label: str,
    first_test: str,
    models: Mapping[str, ModelFactory | tuple[ModelFactory, list[str]]],
    rank_baselines: Sequence[str] = ("pts", "pie"),
) -> pd.DataFrame:
    """Out-of-sample predictions: every season from ``first_test`` on is scored by models fit
    only on strictly earlier seasons. Returns one row per (test season, candidate).

    A model is a factory, or a ``(factory, features)`` pair to use its own feature subset."""
    seasons = sorted(panels)
    frames = []
    for season in (s for s in seasons if s >= first_test):
        train = pd.concat([panels[s] for s in seasons if s < season], ignore_index=True)
        test = panels[season].copy()
        for name in rank_baselines:
            test[f"score_rank_{name}"] = test[name].to_numpy(dtype=float)
        y_train = train[label].to_numpy(dtype=int)
        for name, spec in models.items():
            factory, own = spec if isinstance(spec, tuple) else (spec, features)
            model = factory().fit(as_matrix(train, own), y_train)
            test[f"p_{name}"] = model.predict_proba(as_matrix(test, own))[:, 1]
        test["test_season"] = season
        frames.append(test)
    return pd.concat(frames, ignore_index=True)


def evaluate(pred: pd.DataFrame, label: str, prob_models: Sequence[str], rank_cols: Sequence[str]):
    """Per season R-precision for every score column, plus pooled probability scores and the
    paired log-loss gain of each probabilistic model over the first one (the reference)."""
    score_cols = [f"score_rank_{c}" for c in rank_cols] + [f"p_{m}" for m in prob_models]
    rows = []
    for season, g in pred.groupby("test_season"):
        truth = g[label].to_numpy()
        row = {"season": season, "winners": int(truth.sum())}
        for col in score_cols:
            row[col] = r_precision(g[col].to_numpy(), truth)
        rows.append(row)
    per_season = pd.DataFrame(rows)

    y = pred[label].to_numpy(dtype=int)
    reference = f"p_{prob_models[0]}"
    pooled = {}
    for m in prob_models:
        p = pred[f"p_{m}"].to_numpy()
        row = {"log_loss": log_loss(y, p), "brier": brier_score(y, p)}
        if f"p_{m}" != reference:
            gain, low, high = paired_log_loss_gain(y, p, pred[reference].to_numpy())
            row |= {"gain_vs_reference": gain, "gain_low": low, "gain_high": high}
        pooled[m] = row
    return per_season, pd.DataFrame(pooled).T
