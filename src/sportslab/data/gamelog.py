"""Player game log from NBA.com: one row per player per game played, a whole season per request.

Only games a player actually played appear (no rows for games he missed), so everything built on
this table is "if he plays". NBA.com's FANTASY_PTS equals PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK
- TOV (checked against the raw columns).
"""

from collections.abc import Sequence
from datetime import timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from sportslab.data.client import NbaClient

GAMELOG_COLUMNS = [
    "season",
    "game_id",
    "game_date",
    "player_id",
    "player",
    "team_id",
    "team",
    "opponent",
    "home",
    "minutes",
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "fantasy",
]


def fantasy_points(pts, reb, ast, stl, blk, tov):
    """Standard fantasy score (the one NBA.com publishes)."""
    return pts + 1.2 * reb + 1.5 * ast + 3 * stl + 3 * blk - tov


def clean_gamelog(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Rename, parse the matchup into opponent and home/away, and check the fantasy column."""
    if raw.empty:
        return pd.DataFrame(columns=GAMELOG_COLUMNS)
    matchup = raw["MATCHUP"]
    home = matchup.str.contains(" vs. ", regex=False)
    out = pd.DataFrame(
        {
            "season": season,
            "game_id": raw["GAME_ID"],
            "game_date": pd.to_datetime(raw["GAME_DATE"]),
            "player_id": raw["PLAYER_ID"],
            "player": raw["PLAYER_NAME"],
            "team_id": raw["TEAM_ID"],
            "team": raw["TEAM_ABBREVIATION"],
            "opponent": matchup.str.split(r" vs\. | @ ", regex=True).str[-1],
            "home": home,
            "minutes": raw["MIN"].astype(float),
            "pts": raw["PTS"],
            "reb": raw["REB"],
            "ast": raw["AST"],
            "stl": raw["STL"],
            "blk": raw["BLK"],
            "tov": raw["TOV"],
            "fantasy": raw["FANTASY_PTS"].astype(float),
        }
    )
    calc = fantasy_points(out["pts"], out["reb"], out["ast"], out["stl"], out["blk"], out["tov"])
    if ((calc - out["fantasy"]).abs() > 0.15).mean() > 0.01:
        raise ValueError("FANTASY_PTS no longer matches the standard formula; check the endpoint.")
    return out.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def load_player_games(
    client: NbaClient, seasons: Sequence[str], max_age: timedelta | None = None
) -> pd.DataFrame:
    """Regular-season player games. ``max_age`` for the season in progress (e.g. 6 hours)."""
    frames = []
    for season in seasons:
        (raw,) = client.fetch(
            LeagueGameLog,
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
            max_age=max_age,
        )
        frames.append(clean_gamelog(raw, season))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GAMELOG_COLUMNS)
