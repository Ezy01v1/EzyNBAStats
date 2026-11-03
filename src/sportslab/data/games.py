"""Game-level table built from NBA.com team game logs (LeagueGameLog)."""

from collections.abc import Sequence
from datetime import timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from sportslab.data.client import NbaClient

GAME_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_pts",
    "away_pts",
    "home_win",
    "neutral_site",
]


def team_logs_to_games(log: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per game (not per team), sorted chronologically.

    LeagueGameLog has one row per team per game. The home side is the one whose MATCHUP reads
    "BOS vs. NYK"; the visitor's reads "NYK @ BOS".
    """
    if log.empty:  # e.g. a season that has not started yet
        return pd.DataFrame(columns=GAME_COLUMNS)
    is_home = log["MATCHUP"].str.contains(" vs. ", regex=False)
    by_game = log["GAME_ID"]
    n_home_rows = is_home.groupby(by_game).transform("sum")
    if (n_home_rows > 1).any():
        raise ValueError("Found a game where both teams are marked as home.")
    # Neutral-site games (e.g. played abroad) mark BOTH teams "@": there is no real home team.
    # Keep the game (it is real information), flag it, and label the lower TEAM_ID as "home".
    neutral = n_home_rows == 0
    is_home = is_home | (
        neutral & (log["TEAM_ID"] == log["TEAM_ID"].groupby(by_game).transform("min"))
    )
    neutral_ids = set(log.loc[neutral, "GAME_ID"])
    cols = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]
    home = log.loc[is_home, [*cols, "GAME_DATE"]].rename(
        columns={
            "TEAM_ID": "home_team_id",
            "TEAM_ABBREVIATION": "home_team",
            "PTS": "home_pts",
        }
    )
    away = log.loc[~is_home, cols].rename(
        columns={
            "TEAM_ID": "away_team_id",
            "TEAM_ABBREVIATION": "away_team",
            "PTS": "away_pts",
        }
    )
    games = home.merge(away, on="GAME_ID", validate="one_to_one")
    games = games.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = season
    games["neutral_site"] = games["game_id"].isin(neutral_ids)
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
    return games[GAME_COLUMNS].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_games(
    client: NbaClient, seasons: Sequence[str], max_age: timedelta | None = None
) -> pd.DataFrame:
    """Regular-season games for the given seasons (e.g. ``["2023-24", "2024-25"]``).

    Leave ``max_age`` as None for finished seasons (cached forever); pass e.g.
    ``timedelta(hours=6)`` for the season in progress so new games get picked up.
    """
    frames = []
    for season in seasons:
        (log,) = client.fetch(
            LeagueGameLog, season=season, season_type_all_star="Regular Season", max_age=max_age
        )
        frames.append(team_logs_to_games(log, season))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GAME_COLUMNS)


def season_labels(first_start_year: int, last_start_year: int) -> list[str]:
    """``season_labels(2015, 2017)`` -> ``["2015-16", "2016-17", "2017-18"]``."""
    return [f"{y}-{str(y + 1)[2:]}" for y in range(first_start_year, last_start_year + 1)]
