"""Player comparison: eligible pool, percentile profiles and rank-association helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

MIN_MINUTES = 1000  # ~40 games at 25 min: below this, rate stats are too noisy to rank
RADAR_METRICS = {
    "ts_pct": "TS%",
    "usg_pct": "USG%",
    "ast_pct": "AST%",
    "reb_pct": "REB%",
    "net_rating": "Net rating",
    "pie": "PIE",
}
N_BOOT = 2000


def eligible_pool(players: pd.DataFrame, min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    """Players with at least ``min_minutes`` total minutes: the population percentiles refer to."""
    return players[players["minutes"] >= min_minutes].reset_index(drop=True)


def percentile_profile(
    players: pd.DataFrame, pool: pd.DataFrame, metrics: Sequence[str] = tuple(RADAR_METRICS)
) -> pd.DataFrame:
    """Percentile (0-100) of each player's metrics against ``pool``, ties at the mid-rank.

    A value equal to k pool values counts half of them as "below", so the best of 100 distinct
    players gets 99.5 rather than 100 and a pool member compared with itself is never penalized.
    ``players`` need not be in the pool (a bench player can be placed against the rotation).
    """
    out = pd.DataFrame(index=players.index)
    n = len(pool)
    for metric in metrics:
        ref = np.sort(pool[metric].to_numpy(dtype=float))
        x = players[metric].to_numpy(dtype=float)
        below = np.searchsorted(ref, x, side="left")
        at_or_below = np.searchsorted(ref, x, side="right")
        out[metric] = 100 * (below + at_or_below) / (2 * n)
    return out


def _ranks(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def _residuals(y: np.ndarray, control: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(control, y, 1)
    return y - (slope * control + intercept)


def _spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray | None = None) -> float:
    rx, ry = _ranks(x), _ranks(y)
    if control is not None:
        rc = _ranks(control)
        rx, ry = _residuals(rx, rc), _residuals(ry, rc)
    return float(np.corrcoef(rx, ry)[0, 1])


def rank_correlation(
    x: pd.Series,
    y: pd.Series,
    control: pd.Series | None = None,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Spearman correlation with a 95% bootstrap interval (players resampled).

    With ``control``, returns the *partial* rank correlation: the association left after removing
    what the control's ranks explain linearly from both variables.
    """
    xs, ys = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    cs = None if control is None else control.to_numpy(dtype=float)
    estimate = _spearman(xs, ys, cs)
    rng = np.random.default_rng(seed)
    n = len(xs)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = _spearman(xs[idx], ys[idx], None if cs is None else cs[idx])
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return estimate, float(low), float(high)


def correlation_direction(low: float, high: float) -> str:
    """ "positive", "negative" or "none": whether the whole interval sits on one side of 0."""
    if low > 0:
        return "positive"
    if high < 0:
        return "negative"
    return "none"


def top_overlap(pool: pd.DataFrame, a: str, b: str, n: int = 20) -> tuple[int, set[str]]:
    """How many players appear in both top-``n`` lists (by metric ``a`` and by metric ``b``)."""
    top_a = set(pool.nlargest(n, a)["player"])
    top_b = set(pool.nlargest(n, b)["player"])
    both = top_a & top_b
    return len(both), both


def binned_means(pool: pd.DataFrame, x: str, y: str, bins: int = 5) -> pd.DataFrame:
    """Mean of ``y`` within equal-count bins of ``x`` (n, mean, 95% interval of the mean)."""
    groups = pd.qcut(pool[x], bins, duplicates="drop")
    grouped = pool.groupby(groups, observed=True)
    out = pd.DataFrame(
        {
            f"{x}_mean": grouped[x].mean(),
            "n": grouped[y].size(),
            f"{y}_mean": grouped[y].mean(),
            "sd": grouped[y].std(),
        }
    )
    half = 1.96 * out["sd"] / np.sqrt(out["n"])
    out["ci_low"] = out[f"{y}_mean"] - half
    out["ci_high"] = out[f"{y}_mean"] + half
    return out.drop(columns="sd").reset_index(drop=True)
