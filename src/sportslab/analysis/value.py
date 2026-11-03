"""Market model of salary vs production, and who is paid below / above it.

Production of a player-season is ``contribution = PIE x minutes`` (the share of game events he
accounts for, weighted by how long he plays). The "market" is what a regression of log salary on
log contribution (and age) says players with that production are paid, so a player's gap to it is
a statement about the market, NOT about his true worth. Everything here is observational.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sportslab.analysis.comparator import rank_correlation

MIN_MINUTES = 1000


def contribution(players: pd.DataFrame) -> pd.Series:
    """PIE x total minutes (a production measure in "events share x minutes")."""
    return players["pie"] * players["minutes"]


def design(players: pd.DataFrame, rookie_control: bool = True) -> np.ndarray:
    """Columns of the market regression: log contribution, age, age squared and, when available,
    a rookie-contract flag (rookie scale underpays by the collective agreement, not the market)."""
    x = pd.DataFrame(
        {
            "log_contribution": np.log(contribution(players).clip(lower=1e-6)),
            "age": players["age"],
            "age2": players["age"] ** 2,
        }
    )
    if rookie_control:
        if "years_experience" in players and players["years_experience"].notna().all():
            x["rookie_deal"] = (players["years_experience"] <= 3).astype(float)
        else:
            x["rookie_deal"] = (players["age"] <= 22).astype(float)  # crude proxy, stated in UI
    return x.to_numpy(dtype=float)


def fit_market(players: pd.DataFrame, rookie_control: bool = True) -> LinearRegression:
    """OLS of log salary on production, age and the rookie flag, over rotation players."""
    pool = players[players["minutes"] >= MIN_MINUTES]
    return LinearRegression().fit(design(pool, rookie_control), np.log(pool["salary_usd"]))


def score_players(
    players: pd.DataFrame, rookie_control: bool = True, n_boot: int = 500, seed: int = 0
) -> pd.DataFrame:
    """For each rotation player: market salary, ratio actual/market and a bootstrap interval.

    ``ratio`` < 1 means paid below the market for his production ("undervalued"), > 1 above. The
    interval comes from refitting the market on resampled players, so it captures how uncertain
    the market line is, but NOT the noise in PIE itself (a hot year looks undervalued). ``gap_z``
    compares the gap with the usual spread of gaps (|z| > 2 = unusual even among comparable
    players); it is the better guide to "who is far from the market".
    """
    pool = players[players["minutes"] >= MIN_MINUTES].reset_index(drop=True)
    x, y = design(pool, rookie_control), np.log(pool["salary_usd"].to_numpy())
    model = LinearRegression().fit(x, y)
    fitted = model.predict(x)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(pool)))
    for b in range(n_boot):
        idx = rng.integers(0, len(pool), size=len(pool))
        draws[b] = LinearRegression().fit(x[idx], y[idx]).predict(x)
    out = pool.copy()
    out["market_salary"] = np.exp(fitted)
    out["log_gap"] = y - fitted
    out["ratio"] = np.exp(out["log_gap"])
    # How unusual the gap is next to the typical spread of gaps (residual sd, df-corrected).
    out["gap_z"] = out["log_gap"] / float(np.std(out["log_gap"], ddof=x.shape[1] + 1))
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    out["ratio_low"] = np.exp(y - high)
    out["ratio_high"] = np.exp(y - low)
    return out


def market_fit_summary(players: pd.DataFrame, rookie_control: bool = True) -> dict[str, float]:
    """R-squared and the elasticity of salary to production (log-log slope)."""
    pool = players[players["minutes"] >= MIN_MINUTES]
    x, y = design(pool, rookie_control), np.log(pool["salary_usd"].to_numpy())
    model = LinearRegression().fit(x, y)
    return {
        "n": float(len(pool)),
        "r2": float(model.score(x, y)),
        "elasticity": float(model.coef_[0]),
    }


def gap_persistence(
    before: pd.DataFrame, after: pd.DataFrame, rookie_control: bool = True, n_boot: int = 1000
) -> tuple[float, float, float, int]:
    """Do players paid below the market stay below it next season? Spearman between the log gaps
    of two consecutive seasons (same players), with a bootstrap 95% interval and the count."""
    a = score_players(before, rookie_control, n_boot=1)[["player_id", "log_gap"]]
    b = score_players(after, rookie_control, n_boot=1)[["player_id", "log_gap"]]
    both = a.merge(b, on="player_id", suffixes=("_1", "_2"))
    rho, low, high = rank_correlation(both["log_gap_1"], both["log_gap_2"], n_boot=n_boot)
    return rho, low, high, len(both)
