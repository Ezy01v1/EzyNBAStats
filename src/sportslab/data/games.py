"""Game-level table built from NBA.com team game logs (LeagueGameLog)."""

from collections.abc import Sequence

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
]


def team_logs_to_games(log: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per game (not per team), sorted chronologically.

    LeagueGameLog has one row per team per game. The home side is the one whose MATCHUP reads
    "BOS vs. NYK"; the visitor's reads "NYK @ BOS".
    """
    is_home = log["MATCHUP"].str.contains(" vs. ", regex=False)
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
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
    return games[GAME_COLUMNS].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_games(client: NbaClient, seasons: Sequence[str]) -> pd.DataFrame:
    """Regular-season games for the given seasons (e.g. ``["2023-24", "2024-25"]``)."""
    frames = []
    for season in seasons:
        (log,) = client.fetch(LeagueGameLog, season=season, season_type_all_star="Regular Season")
        frames.append(team_logs_to_games(log, season))
    return pd.concat(frames, ignore_index=True)


def season_labels(first_start_year: int, last_start_year: int) -> list[str]:
    """``season_labels(2015, 2017)`` -> ``["2015-16", "2016-17", "2017-18"]``."""
    return [f"{y}-{str(y + 1)[2:]}" for y in range(first_start_year, last_start_year + 1)]
