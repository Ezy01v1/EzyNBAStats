"""Scoring rules for probabilistic win predictions. Lower log loss / Brier is better."""

import numpy as np
import pandas as pd

_EPS = 1e-15


def log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log-likelihood. Punishes confident wrong answers very hard."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability. 0.25 is what a constant 50% scores."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def accuracy(y_true: np.ndarray, p: np.ndarray) -> float:
    """Share of games where the favored side (p > .5) won. Ignores how confident we were."""
    y = np.asarray(y_true)
    return float(np.mean((np.asarray(p) > 0.5).astype(int) == y))


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per probability bin: games, mean predicted probability, observed win rate.

    A calibrated model has ``mean_pred`` close to ``observed`` in every bin. Empty bins are
    dropped; small ``n`` means the row is noisy.
    """
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "y": np.asarray(y_true, dtype=float), "p": p})
    table = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    return table.reset_index()


def score_all(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(y_true, p),
        "brier": brier_score(y_true, p),
        "accuracy": accuracy(y_true, p),
    }
