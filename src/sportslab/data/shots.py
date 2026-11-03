"""Shot-level data from NBA.com (ShotChartDetail): one player-season of attempts, plus the
league-wide made/attempted counts per zone for that season (the same for every player)."""

from datetime import timedelta

import pandas as pd
from nba_api.stats.endpoints import ShotChartDetail

from sportslab.data.client import NbaClient

SHOT_RENAMES = {
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "PLAYER_ID": "player_id",
    "PERIOD": "period",
    "ACTION_TYPE": "action_type",
    "SHOT_TYPE": "shot_type",
    "SHOT_ZONE_BASIC": "zone_basic",
    "SHOT_ZONE_AREA": "zone_area",
    "SHOT_ZONE_RANGE": "zone_range",
    "SHOT_DISTANCE": "distance",
    "LOC_X": "x",
    "LOC_Y": "y",
    "SHOT_MADE_FLAG": "made",
}
SHOT_COLUMNS = [*SHOT_RENAMES.values(), "points"]
LEAGUE_COLUMNS = ["zone_basic", "zone_area", "zone_range", "fga", "fgm"]


def clean_shots(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and add ``points`` = value of the shot if made (2 or 3, free throws aside)."""
    if raw.empty:
        return pd.DataFrame(columns=SHOT_COLUMNS)
    if not (raw["SHOT_ATTEMPTED_FLAG"] == 1).all():
        raise ValueError("Unexpected rows with SHOT_ATTEMPTED_FLAG != 1 in a FGA shot chart.")
    out = raw[list(SHOT_RENAMES)].rename(columns=SHOT_RENAMES)
    out["game_date"] = pd.to_datetime(out["game_date"], format="%Y%m%d")
    out["points"] = out["shot_type"].str.startswith("3").map({True: 3, False: 2})
    return out.reset_index(drop=True)


def clean_league_zones(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.rename(
        columns={
            "SHOT_ZONE_BASIC": "zone_basic",
            "SHOT_ZONE_AREA": "zone_area",
            "SHOT_ZONE_RANGE": "zone_range",
            "FGA": "fga",
            "FGM": "fgm",
        }
    )
    return out[LEAGUE_COLUMNS].reset_index(drop=True)


def load_player_shots(
    client: NbaClient, player_id: int, season: str, max_age: timedelta | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regular-season attempts of one player (``team_id=0`` covers traded players) and the
    league-wide zone totals for the season. One request per player-season."""
    raw_shots, raw_league = client.fetch(
        ShotChartDetail,
        player_id=player_id,
        team_id=0,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
        max_age=max_age,
    )
    return clean_shots(raw_shots), clean_league_zones(raw_league)
