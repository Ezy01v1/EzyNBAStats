"""Pre-game features: everything here is known *before* tip-off.

For game i, a feature may only use games that happened strictly before game i. Every rolling
quantity is shifted by one game for exactly that reason (tests flip a game's result and check
that its own features do not move).
"""

from __future__ import annotations

import pandas as pd

from sportslab.models.elo import EloParams, run_elo

FORM_WINDOW = 10  # last N games of the same season
FORM_PRIOR_GAMES = 5  # shrinkage: form = sum(margins) / (games_in_window + FORM_PRIOR_GAMES)
MAX_REST_DAYS = 7  # a 4-month offseason is not "207 days of rest"


def _team_game_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with the margin from that team's point of view."""
    margin = games["home_pts"] - games["away_pts"]
    common = {"row": games.index, "game_date": games["game_date"], "season": games["season"]}
    home = pd.DataFrame(
        {**common, "team_id": games["home_team_id"], "margin": margin, "side": "home"}
    )
    away = pd.DataFrame(
        {**common, "team_id": games["away_team_id"], "margin": -margin, "side": "away"}
    )
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team_id", "game_date", "row"], kind="stable").reset_index(drop=True)


def _add_rest_and_form(long: pd.DataFrame) -> pd.DataFrame:
    days_since_last = long.groupby("team_id")["game_date"].diff().dt.days
    long["rest"] = days_since_last.fillna(MAX_REST_DAYS).clip(upper=MAX_REST_DAYS)

    # Margin of the previous games of the same team *in the same season* (shift(1) = strictly past).
    season_group = long.groupby(["team_id", "season"])["margin"]
    long["_past_margin"] = season_group.shift(1)
    window = long.groupby(["team_id", "season"])["_past_margin"]
    total = window.transform(lambda s: s.rolling(FORM_WINDOW, min_periods=0).sum())
    count = window.transform(lambda s: s.rolling(FORM_WINDOW, min_periods=0).count())
    long["form"] = total / (count + FORM_PRIOR_GAMES)
    return long


def build_pregame_features(games: pd.DataFrame, elo_params: EloParams) -> pd.DataFrame:
    """Return ``games`` plus pre-game features (home minus away where it makes sense).

    Needs chronological games with game_date, season, team ids, points, home_win, neutral_site.
    Columns added: elo_home_pre, elo_away_pre, p_elo, elo_diff, home/away_rest, home/away_form,
    home/away_b2b, form_diff, rest_diff.
    """
    games = games.reset_index(drop=True)
    long = _add_rest_and_form(_team_game_table(games))
    home = long[long["side"] == "home"].set_index("row")
    away = long[long["side"] == "away"].set_index("row")

    out = run_elo(games, elo_params).rename(columns={"p_home": "p_elo"})
    out["elo_diff"] = out["elo_home_pre"] - out["elo_away_pre"]
    out["home_rest"] = home["rest"].reindex(out.index).to_numpy()
    out["away_rest"] = away["rest"].reindex(out.index).to_numpy()
    out["home_form"] = home["form"].reindex(out.index).to_numpy()
    out["away_form"] = away["form"].reindex(out.index).to_numpy()
    out["home_b2b"] = (out["home_rest"] == 1).astype(int)
    out["away_b2b"] = (out["away_rest"] == 1).astype(int)
    out["form_diff"] = out["home_form"] - out["away_form"]
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    return out
