"""Lineup net ratings with honest uncertainty, and a test of whether they persist.

Net rating of a lineup = points scored minus allowed per 100 possessions. Over a few hundred
possessions it is mostly noise, and the lineups that played together are not independent of the
team they belong to, so we ask (1) how big is the sampling error, (2) do ratings survive from
the first half of a season to the second, and (3) is the lineup telling us more than its team's
rating already does.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sportslab.analysis.comparator import rank_correlation

POSS_PER_GAME = 100.0  # a team plays ~100 possessions per game


def margin_sd_per_game(games: pd.DataFrame) -> float:
    """Spread of final margins in points, from the game table (includes real team differences, so
    it slightly overstates pure sampling noise: the intervals below err on the wide side)."""
    return float((games["home_pts"] - games["away_pts"]).std())


def net_rating_se(poss: pd.Series | np.ndarray, margin_sd: float) -> pd.Series | np.ndarray:
    """Standard error of a net rating (per 100 possessions) measured over ``poss`` possessions.

    A game of ~100 possessions has a margin with spread ``margin_sd``; the spread of the average
    over n possessions shrinks with the square root of n.
    """
    return margin_sd * np.sqrt(POSS_PER_GAME / np.asarray(poss, dtype=float))


def add_uncertainty(lineups: pd.DataFrame, margin_sd: float) -> pd.DataFrame:
    """Adds ``se``, ``ci_low`` and ``ci_high`` (95%) around each lineup's net rating."""
    out = lineups.copy()
    out["se"] = net_rating_se(out["poss"], margin_sd)
    out["ci_low"] = out["net_rating"] - 1.96 * out["se"]
    out["ci_high"] = out["net_rating"] + 1.96 * out["se"]
    return out


def shrink(net: pd.Series, poss: pd.Series, prior: pd.Series | float, k: float) -> pd.Series:
    """(net * poss + prior * k) / (poss + k): pull a rating towards ``prior`` as if it had been
    observed over ``k`` extra possessions."""
    return (net * poss + prior * k) / (poss + k)


def paired_halves(
    first: pd.DataFrame, second: pd.DataFrame, team_first: pd.DataFrame, min_poss: float
) -> pd.DataFrame:
    """Lineups seen in both halves with at least ``min_poss`` possessions in each.

    Columns: net_1, poss_1, net_2, poss_2 and ``team_net_1`` (the team's first-half net rating,
    known at mid-season). Selecting lineups that kept playing in both halves is itself a filter:
    coaches keep lineups that look good, so persistence here is likely optimistic.
    """
    cols = ["lineup_id", "players", "team_id", "net_rating", "poss"]
    both = first[cols].merge(
        second[["lineup_id", "team_id", "net_rating", "poss"]],
        on=["lineup_id", "team_id"],
        suffixes=("_1", "_2"),
    )
    both = both[(both["poss_1"] >= min_poss) & (both["poss_2"] >= min_poss)]
    both = both.merge(
        team_first[["team_id", "net_rating"]].rename(columns={"net_rating": "team_net_1"}),
        on="team_id",
        validate="many_to_one",
    )
    return both.rename(columns={"net_rating_1": "net_1", "net_rating_2": "net_2"}).reset_index(
        drop=True
    )


def persistence(joined: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    """Spearman correlation (with 95% bootstrap interval) between first- and second-half ratings:
    of the lineup itself, and of the part beyond its team (lineup minus team net rating)."""
    raw = rank_correlation(joined["net_1"], joined["net_2"], n_boot=n_boot, seed=seed)
    beyond = rank_correlation(
        joined["net_1"] - joined["team_net_1"],
        joined["net_2"] - joined["team_net_1"],
        n_boot=n_boot,
        seed=seed,
    )
    return {"raw": raw, "beyond_team": beyond}


def prediction_errors(joined: pd.DataFrame, ks: Sequence[float], prior: str = "team") -> pd.Series:
    """Possession-weighted mean squared error of predicting the second-half net rating.

    Prediction = first-half rating shrunk towards ``prior`` ("team": the team's first-half net
    rating; "zero": league average) with ``k`` pseudo-possessions. k = 0 is the raw rating;
    a huge k is the prior alone. The second-half rating is noisy, but that noise is the same for
    every predictor, so a lower error still means a better one.
    """
    target = joined["net_2"]
    weights = joined["poss_2"]
    prior_value = joined["team_net_1"] if prior == "team" else 0.0
    errors = {}
    for k in ks:
        pred = shrink(joined["net_1"], joined["poss_1"], prior_value, k)
        errors[k] = float((weights * (target - pred) ** 2).sum() / weights.sum())
    return pd.Series(errors, name=f"error (prior={prior})")


def paired_error_gain(
    joined: pd.DataFrame,
    k_ref: float,
    k_new: float,
    prior: str = "team",
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Possession-weighted drop in squared error of ``k_new`` vs ``k_ref`` (positive = k_new is
    better), with a 95% bootstrap interval over lineups. Both predictors see the same lineups."""
    prior_value = joined["team_net_1"] if prior == "team" else 0.0
    target, w = joined["net_2"].to_numpy(), joined["poss_2"].to_numpy()

    def squared_error(k):
        pred = shrink(joined["net_1"], joined["poss_1"], prior_value, k).to_numpy()
        return (target - pred) ** 2

    diff = squared_error(k_ref) - squared_error(k_new)
    rng = np.random.default_rng(seed)
    n = len(diff)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = (w[idx] * diff[idx]).sum() / w[idx].sum()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float((w * diff).sum() / w.sum()), float(low), float(high)
