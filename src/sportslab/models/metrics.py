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
    """Share of games where the favored side (p > .5) won. Ignores how confident we were.
    A dead-even prediction (p == .5) names no favorite, so it counts as half right."""
    y = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    hit = np.where(p == 0.5, 0.5, ((p > 0.5).astype(int) == y).astype(float))
    return float(np.mean(hit))


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per probability bin: games, mean predicted probability, observed win rate.

    A calibrated model has ``mean_pred`` close to ``observed`` in every bin. Empty bins are
    dropped. ``ci_low``/``ci_high`` is a 95% Wilson interval for the observed rate: with few
    games per bin it is wide, and a gap inside it is just noise.
    """
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "y": np.asarray(y_true, dtype=float), "p": p})
    table = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    z = 1.96
    n, rate = table["n"], table["observed"]
    denom = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denom
    table["ci_low"] = center - half
    table["ci_high"] = center + half
    return table.reset_index()


def score_all(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(y_true, p),
        "brier": brier_score(y_true, p),
        "accuracy": accuracy(y_true, p),
    }
