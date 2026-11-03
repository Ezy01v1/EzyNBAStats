"""Building blocks of the daily dashboard: what happened, what is next, and where teams stand.

Everything here is descriptive or a probability from the Elo model of Module 1; nothing knows
about injuries, rest or lineups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportslab.models.season_sim import win_prob


def last_game_day(schedule: pd.DataFrame) -> pd.Timestamp | None:
    """The most recent date with a finished game."""
    played = schedule.loc[schedule["played"], "game_date"]
    return None if played.empty else played.max().normalize()


def next_game_day(schedule: pd.DataFrame, today: pd.Timestamp) -> pd.Timestamp | None:
    """The first date on or after ``today`` with an unplayed game."""
    pending = schedule.loc[~schedule["played"] & (schedule["game_date"] >= today.normalize())]
    return None if pending.empty else pending["game_date"].min().normalize()


def games_on(schedule: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return schedule[schedule["game_date"].dt.normalize() == day.normalize()].reset_index(drop=True)


def add_probabilities(
    games: pd.DataFrame, ratings: dict[int, float], home_advantage: float, initial: float = 1500.0
) -> pd.DataFrame:
    """Elo probability that the home side wins (neutral-site games get no home bonus)."""
    out = games.copy()
    r_home = out["home_team_id"].map(lambda t: ratings.get(t, initial))
    r_away = out["away_team_id"].map(lambda t: ratings.get(t, initial))
    bonus = np.where(out["neutral_site"], 0.0, home_advantage)
    out["p_home"] = win_prob(r_home.to_numpy(), r_away.to_numpy(), bonus)
    return out


def surprises(results: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Finished games won by a side that the Elo model gave at most ``threshold`` pre-game."""
    home_won = (results["home_pts"] > results["away_pts"]).to_numpy()
    p_winner = np.where(home_won, results["p_home"].to_numpy(), 1 - results["p_home"].to_numpy())
    mask = (p_winner <= threshold) & results["played"].to_numpy()
    return results[mask].assign(p_winner=p_winner[mask])


def standings(played: pd.DataFrame, conferences: pd.DataFrame, last_n: int = 10) -> pd.DataFrame:
    """Wins, losses, win%, record over the last ``last_n`` games and mean point difference.

    ``played``: finished games (home_team_id, away_team_id, home_pts, away_pts, game_date).
    Teams with no games yet are listed with zeros.
    """
    home = pd.DataFrame(
        {
            "team_id": played["home_team_id"],
            "game_date": played["game_date"],
            "win": played["home_pts"] > played["away_pts"],
            "diff": played["home_pts"] - played["away_pts"],
        }
    )
    away = pd.DataFrame(
        {
            "team_id": played["away_team_id"],
            "game_date": played["game_date"],
            "win": played["away_pts"] > played["home_pts"],
            "diff": played["away_pts"] - played["home_pts"],
        }
    )
    long = pd.concat([home, away]).sort_values(["team_id", "game_date"])
    grouped = long.groupby("team_id")
    table = grouped.agg(wins=("win", "sum"), games=("win", "size"), diff=("diff", "mean"))
    table["losses"] = table["games"] - table["wins"]
    table["win_pct"] = table["wins"] / table["games"]
    recent = grouped.tail(last_n).groupby("team_id")["win"].agg(["sum", "size"])
    wins_recent = recent["sum"].astype(int)
    table["last_n"] = wins_recent.astype(str) + "-" + (recent["size"] - wins_recent).astype(str)
    table = table.reset_index().merge(
        conferences[["team_id", "team", "conference"]], on="team_id", how="right"
    )
    table[["wins", "games", "losses"]] = table[["wins", "games", "losses"]].fillna(0).astype(int)
    table["win_pct"] = table["win_pct"].fillna(0.0)
    return table.sort_values(
        ["conference", "win_pct", "diff"], ascending=[True, False, False]
    ).reset_index(drop=True)


def top_performances(player_games: pd.DataFrame, day: pd.Timestamp, n: int = 10) -> pd.DataFrame:
    """Best fantasy-point games of the day (with the box-score line)."""
    d = player_games[player_games["game_date"].dt.normalize() == day.normalize()]
    return d.nlargest(n, "fantasy")[
        [
            "player",
            "team",
            "opponent",
            "minutes",
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "fantasy",
        ]
    ].reset_index(drop=True)
