"""Backtest the season simulator on finished seasons, using only what was known at the cutoff.

At each cutoff the Elo ratings come from games before it, the standings from games played so far,
and the remaining schedule is the real one. The simulator's statements (win intervals, playoff and
title probabilities) are then compared with what actually happened.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sportslab.models.elo import EloParams, final_ratings
from sportslab.models.season_sim import SimConfig, simulate_season

INTERVALS = {
    "50%": ("wins_p25", "wins_p75"),
    "80%": ("wins_p10", "wins_p90"),
    "90%": ("wins_p05", "wins_p95"),
}
GAMES_PER_TEAM = 82


def backtest_season(
    games: pd.DataFrame,
    season: str,
    fractions: Sequence[float],
    conferences: pd.DataFrame,
    outcome: dict,
    elo_params: EloParams,
    n_sims: int = 5000,
    seed: int = 0,
    rating_sd: float = 0.0,
) -> pd.DataFrame:
    """One row per (cutoff, team) with the simulator's output and the truth.

    ``games``: chronological finished games of every season up to ``season`` (needs season,
    home_team_id, away_team_id, home_win, neutral_site). ``outcome``: ``playoff_outcomes`` of the
    season. Extra columns: ``actual_wins``, ``made_playoffs``, ``is_champion``, ``naive_wins``
    (current win percentage extrapolated to 82 games) and ``top8`` (in the top 8 of its conference
    by wins so far).
    """
    season_games = games[games["season"] == season].reset_index(drop=True)
    before = games[games["season"] != season]
    before = before[before["season"] < season]
    actual = _wins(season_games)
    frames = []
    for i, f in enumerate(fractions):
        cut = int(round(f * len(season_games)))
        played, remaining = season_games.iloc[:cut], season_games.iloc[cut:]
        ratings = final_ratings(pd.concat([before, played], ignore_index=True), elo_params)
        current = _wins(played)
        sim = simulate_season(
            ratings,
            conferences,
            current.to_dict(),
            remaining,
            config=SimConfig(
                home_advantage=elo_params.home_advantage,
                n_sims=n_sims,
                seed=seed + i,
                rating_sd=rating_sd,
            ),
        ).table
        sim["cutoff"] = f
        sim["rating_sd"] = rating_sd
        sim["games_played"] = cut
        sim["actual_wins"] = actual.reindex(sim.index).fillna(0)
        sim["made_playoffs"] = sim.index.isin(outcome["made_playoffs"])
        sim["is_champion"] = sim.index == outcome["champion"]
        played_per_team = _games_played(played).reindex(sim.index).fillna(0)
        sim["naive_wins"] = np.where(
            played_per_team > 0,
            sim["current_wins"] / played_per_team.replace(0, 1) * GAMES_PER_TEAM,
            41,
        )
        conf = conferences.set_index("team_id")["conference"].reindex(sim.index)
        sim["conference"] = conf
        sim["top8"] = (
            sim.groupby("conference")["current_wins"].rank(ascending=False, method="first") <= 8
        )
        sim["season"] = season
        frames.append(sim.reset_index())
    return pd.concat(frames, ignore_index=True)


def _wins(games: pd.DataFrame) -> pd.Series:
    home = games.loc[games["home_win"] == 1, "home_team_id"]
    away = games.loc[games["home_win"] == 0, "away_team_id"]
    return pd.concat([home, away]).value_counts()


def _games_played(games: pd.DataFrame) -> pd.Series:
    return pd.concat([games["home_team_id"], games["away_team_id"]]).value_counts()


def summarize_backtest(frame: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """Per (season, cutoff): interval coverage, win errors, playoff Brier and title log score."""
    rows = []
    keys = [c for c in ("rating_sd", "season", "cutoff") if c in frame.columns]
    for values, g in frame.groupby(keys):
        row = dict(zip(keys, values, strict=True))
        for name, (low, high) in INTERVALS.items():
            row[f"coverage_{name}"] = float(
                ((g["actual_wins"] >= g[low]) & (g["actual_wins"] <= g[high])).mean()
            )
        row["mae_wins_sim"] = float((g["exp_wins"] - g["actual_wins"]).abs().mean())
        row["mae_wins_naive"] = float((g["naive_wins"] - g["actual_wins"]).abs().mean())
        y = g["made_playoffs"].astype(float)
        row["brier_playoffs_sim"] = float(((g["p_playoffs"] - y) ** 2).mean())
        row["brier_playoffs_top8"] = float(((g["top8"].astype(float) - y) ** 2).mean())
        row["brier_playoffs_const"] = float(((y.mean() - y) ** 2).mean())
        champ = g[g["is_champion"]]
        p = float(champ["p_champion"].iloc[0]) if len(champ) else float("nan")
        row["p_champion_actual"] = p
        row["logscore_title"] = float(-np.log(max(p, 0.5 / n_sims)))
        row["champion_rank"] = int(
            (g["p_champion"] > (champ["p_champion"].iloc[0] if len(champ) else 0)).sum() + 1
        )
        row["top_title_prob"] = float(g["p_champion"].max())
        rows.append(row)
    return pd.DataFrame(rows)
