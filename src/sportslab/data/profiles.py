"""Per-player style inputs for role clustering: per-100-possession box score and shot zones.

Both come from one league-wide request per season (no per-player calls).
"""

import ast
from datetime import timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerShotLocations, LeagueDashPlayerStats

from sportslab.data.client import NbaClient

PER100_RENAMES = {
    "PLAYER_ID": "player_id",
    "FGA": "fga100",
    "FG3A": "fg3a100",
    "FTA": "fta100",
    "OREB": "oreb100",
    "DREB": "dreb100",
    "AST": "ast100",
    "TOV": "tov100",
    "STL": "stl100",
    "BLK": "blk100",
    "PF": "pf100",
}
# NBA.com's "MIN" in this per-100 table is minutes per 100 possessions (~48), NOT minutes played:
# real minutes come from clean_player_stats (the Advanced table).

ZONE_ATTEMPTS = {
    "Restricted Area": "ra_fga",
    "In The Paint (Non-RA)": "paint_fga",
    "Mid-Range": "mid_fga",
    "Corner 3": "corner3_fga",
    "Above the Break 3": "atb3_fga",
}


def clean_per100(raw: pd.DataFrame) -> pd.DataFrame:
    if raw["PLAYER_ID"].duplicated().any():
        raise ValueError("Expected one row per player in the per-100 table.")
    return raw[list(PER100_RENAMES)].rename(columns=PER100_RENAMES).reset_index(drop=True)


def _flatten_column(column) -> tuple[str, str]:
    """(zone, stat) from a MultiIndex tuple, or from its string form after a Parquet round trip."""
    if isinstance(column, str):
        column = ast.literal_eval(column) if column.startswith("(") else ("", column)
    zone, stat = column
    return zone, stat


def clean_shot_zones(raw: pd.DataFrame) -> pd.DataFrame:
    """Attempts per court zone for each player (``player_id`` + the ``*_fga`` columns).

    The raw table has two-level columns (zone, stat); the cache stores them as strings.
    """
    flat = raw.copy()
    flat.columns = pd.MultiIndex.from_tuples([_flatten_column(c) for c in raw.columns])
    out = pd.DataFrame({"player_id": flat[("", "PLAYER_ID")].to_numpy()})
    for zone, name in ZONE_ATTEMPTS.items():
        # NBA.com leaves NaN (not 0) where a player has no attempt in a zone; seen only for
        # players with a few minutes, but 0 is the correct reading, so say so explicitly.
        out[name] = flat[(zone, "FGA")].fillna(0).to_numpy()
    return out


def load_per100(client: NbaClient, season: str, max_age: timedelta | None = None) -> pd.DataFrame:
    (raw,) = client.fetch(
        LeagueDashPlayerStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Base",
        per_mode_detailed="Per100Possessions",
        max_age=max_age,
    )
    return clean_per100(raw)


def load_shot_zones(
    client: NbaClient, season: str, max_age: timedelta | None = None
) -> pd.DataFrame:
    (raw,) = client.fetch(
        LeagueDashPlayerShotLocations,
        season=season,
        season_type_all_star="Regular Season",
        distance_range="By Zone",
        max_age=max_age,
    )
    return clean_shot_zones(raw)
