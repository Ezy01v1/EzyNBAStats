"""Elo ratings as a win-probability baseline.

Each team has a rating. Before a game, the rating gap (plus a home-court bonus) becomes a win
probability; after it, ratings move by ``K * (result - expected)``. Between seasons ratings are
pulled back toward the league mean because rosters change.

Every prediction is made *before* the game's own result is applied, so there is no look-ahead.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, replace

import pandas as pd

from sportslab.models.metrics import log_loss


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 100.0  # Elo points added to the home team's rating
    carryover: float = 0.75  # share of last season's distance from the mean that is kept
    initial: float = 1500.0


def expected_home_win(r_home: float, r_away: float, home_advantage: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_home + home_advantage - r_away) / 400.0))


def run_elo(games: pd.DataFrame, params: EloParams | None = None) -> pd.DataFrame:
    """Walk through games in order and return them with pre-game ``elo_home_pre``,
    ``elo_away_pre`` and ``p_home`` (probability that the home team wins).

    ``games`` must be chronological and carry season, home_team_id, away_team_id, home_win.
    """
    params = params or EloParams()
    ratings: dict[int, float] = {}
    current_season: str | None = None
    n = len(games)
    elo_home = [0.0] * n
    elo_away = [0.0] * n
    p_home = [0.0] * n

    rows = zip(
        games["season"],
        games["home_team_id"],
        games["away_team_id"],
        games["home_win"],
        strict=True,
    )
    for i, (season, home, away, home_win) in enumerate(rows):
        if season != current_season:
            if current_season is not None:
                for team, rating in ratings.items():
                    ratings[team] = params.initial + params.carryover * (rating - params.initial)
            current_season = season

        r_home = ratings.get(home, params.initial)
        r_away = ratings.get(away, params.initial)
        p = expected_home_win(r_home, r_away, params.home_advantage)
        elo_home[i], elo_away[i], p_home[i] = r_home, r_away, p

        delta = params.k * (home_win - p)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    out = games.copy()
    out["elo_home_pre"] = elo_home
    out["elo_away_pre"] = elo_away
    out["p_home"] = p_home
    return out


def tune_elo(
    games: pd.DataFrame,
    score_seasons: Iterable[str],
    ks: Iterable[float],
    home_advantages: Iterable[float],
    carryovers: Iterable[float],
) -> pd.DataFrame:
    """Grid search scored only on ``score_seasons`` (earlier seasons just warm the ratings up).

    Pass only games up to the validation seasons: the final test season must never be in here.
    Returns one row per parameter combination, best (lowest log loss) first.
    """
    score_seasons = list(score_seasons)
    rows = []
    for k, h, c in itertools.product(ks, home_advantages, carryovers):
        params = replace(EloParams(), k=k, home_advantage=h, carryover=c)
        out = run_elo(games, params)
        scored = out[out["season"].isin(score_seasons)]
        rows.append(
            {
                "k": k,
                "home_advantage": h,
                "carryover": c,
                "log_loss": log_loss(scored["home_win"].to_numpy(), scored["p_home"].to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
