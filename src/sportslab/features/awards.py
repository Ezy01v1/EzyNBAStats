"""Candidate panels for award models: features known at the cutoff, labels from the awards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportslab.data.awards import ALL_NBA, ALL_STAR, selections_before

BASE_FEATURES = [
    "pts",
    "reb",
    "ast",
    "ts_pct",
    "usg_pct",
    "pie",
    "min_pg",
    "gp_frac",
    "team_w_pct",
]
ALL_STAR_FEATURES = [*BASE_FEATURES, "prior_all_stars"]
ALL_NBA_FEATURES = [*BASE_FEATURES, "prior_all_nba"]


def build_panel(
    stats: pd.DataFrame,
    awards: pd.DataFrame,
    labels: pd.DataFrame,
    season: str,
    n_candidates: int = 80,
    min_gp: int = 15,
) -> pd.DataFrame:
    """Candidates of one season: the ``n_candidates`` players with most points per game among
    those with ``min_gp`` games, with features and labels.

    ``gp_frac`` is games played over the most games anyone played so far (a proxy for the share of
    the schedule the player was available). ``prior_all_stars`` / ``prior_all_nba`` count EARLIER
    selections only (never this season's), taken from the players' whole career history.
    """
    s = stats[(stats["season"] == season) & (stats["gp"] >= min_gp)].copy()
    s = s.nlargest(n_candidates, "pts")
    s["gp_frac"] = s["gp"] / stats.loc[stats["season"] == season, "gp"].max()
    for column, award in (("prior_all_stars", ALL_STAR), ("prior_all_nba", ALL_NBA)):
        prior = selections_before(awards, season, award)
        s[column] = s["player_id"].map(prior).fillna(0).astype(int)
    lab = labels[labels["season"] == season].drop(columns="season")
    s = s.merge(lab, on="player_id", how="left")
    for flag in ("all_star", "all_nba", "all_nba_first", "mvp"):
        s[flag] = s[flag].fillna(False).astype(bool)
    return s.reset_index(drop=True)


def as_matrix(panel: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix; count features are log-scaled (the 10th selection is not 10x the 1st)."""
    x = panel[features].copy()
    for c in ("prior_all_stars", "prior_all_nba"):
        if c in x:
            x[c] = np.log1p(x[c])
    return x.to_numpy(dtype=float)
