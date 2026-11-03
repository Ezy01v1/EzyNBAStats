"""Fantasy-point projections with honest intervals, and the baselines they must beat.

Baselines first: the season average to date, then the last 5 games, then a shrunk blend of the
season average and last season's. A ridge regression adds minutes, venue, rest and opponent.
Intervals come from the model's own past errors (empirical quantiles by predicted level), because
a player who averages 40 fantasy points swings more than one who averages 15.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.features.fantasy import FEATURES

SEASON_MEAN_MIN_GAMES = 10
QUANTILES = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
N_BINS = 8


def baseline_season(rows: pd.DataFrame) -> pd.Series:
    """Season average to date once he has 10 games, otherwise last season's average."""
    use_season = rows["n_prior_season"] >= SEASON_MEAN_MIN_GAMES
    return rows["season_mean"].where(use_season, rows["prior_season_mean"])


def baseline_last5(rows: pd.DataFrame) -> pd.Series:
    return rows["last5"]


def shrunk_mean(rows: pd.DataFrame, k: float) -> pd.Series:
    """(sum of this season's points + k x last season's average) / (games + k)."""
    n = rows["n_prior_season"]
    total = rows["season_mean"].fillna(0.0) * n
    return (total + k * rows["prior_season_mean"]) / (n + k)


def fit_shrinkage(train: pd.DataFrame, ks: Sequence[float]) -> float:
    """The ``k`` with the lowest mean absolute error on ``train``."""
    errors = {k: float((train["target"] - shrunk_mean(train, k)).abs().mean()) for k in ks}
    return min(errors, key=errors.get)


def fit_ridge(train: pd.DataFrame, alpha: float = 10.0):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return model.fit(train[FEATURES].to_numpy(dtype=float), train["target"].to_numpy())


def predict_ridge(model, rows: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(rows[FEATURES].to_numpy(dtype=float)), 0.0, None)


class IntervalCalibrator:
    """Empirical prediction intervals: residual quantiles within bins of the predicted value.

    Fit on predictions made out of the sample the point model was fit on (otherwise the residuals
    are too small). The low end is never below 0: nobody scores negative fantasy points here.
    """

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins

    def fit(self, predicted: np.ndarray, actual: np.ndarray) -> IntervalCalibrator:
        predicted = np.asarray(predicted, dtype=float)
        self.edges_ = np.unique(np.quantile(predicted, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        bins = np.searchsorted(self.edges_, predicted, side="right")
        residual = np.asarray(actual, dtype=float) - predicted
        self.quantiles_ = {b: np.quantile(residual[bins == b], QUANTILES) for b in np.unique(bins)}
        return self

    def interval(self, predicted: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bound of the central ``level`` interval (0.5, 0.8 or 0.9)."""
        low_q, high_q = {0.5: (2, 3), 0.8: (1, 4), 0.9: (0, 5)}[level]
        predicted = np.asarray(predicted, dtype=float)
        bins = np.searchsorted(self.edges_, predicted, side="right")
        low = np.empty_like(predicted)
        high = np.empty_like(predicted)
        for i, b in enumerate(bins):
            q = self.quantiles_.get(b, self.quantiles_[max(self.quantiles_)])
            low[i], high[i] = predicted[i] + q[low_q], predicted[i] + q[high_q]
        return np.clip(low, 0.0, None), high


def coverage(actual, low, high) -> float:
    actual = np.asarray(actual)
    return float(((actual >= low) & (actual <= high)).mean())


def paired_abs_error_gain(
    actual: pd.Series,
    new: pd.Series,
    reference: pd.Series,
    players: pd.Series,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Drop in mean absolute error of ``new`` vs ``reference`` (positive = new is better), with a
    95% bootstrap interval that resamples PLAYERS (games of one player are not independent)."""
    diff = (actual - reference).abs() - (actual - new).abs()
    per_player = diff.groupby(players.to_numpy()).agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    n = len(per_player)
    sums, sizes = per_player["sum"].to_numpy(), per_player["size"].to_numpy()
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = sums[idx].sum() / sizes[idx].sum()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
