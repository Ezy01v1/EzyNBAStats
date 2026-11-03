"""Lineup (5-man or 2-man) and team ratings from NBA.com, optionally for a date range.

LeagueDashLineups returns only the ~2000 lineups with the most minutes (verified: 5-man lineups
of 2025-26 stop at 12 minutes and cover about half of all lineup minutes), so conclusions are
about frequently used lineups, never about "all lineups".
"""

from datetime import date, timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueDashLineups, LeagueDashTeamStats

from sportslab.data.client import NbaClient

LINEUP_RENAMES = {
    "GROUP_ID": "lineup_id",
    "GROUP_NAME": "players",
    "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team",
    "GP": "gp",
    "MIN": "minutes",
    "POSS": "poss",
    "OFF_RATING": "off_rating",
    "DEF_RATING": "def_rating",
    "NET_RATING": "net_rating",
}
TEAM_RENAMES = {
    "TEAM_ID": "team_id",
    "TEAM_NAME": "team",
    "GP": "gp",
    "POSS": "poss",
    "NET_RATING": "net_rating",
}


def clean_lineups(raw: pd.DataFrame) -> pd.DataFrame:
    """Select and rename; one row per (lineup, team) with possessions and ratings.

    The same group of players can appear under two teams after a trade, so the key is the
    lineup id together with the team id.
    """
    if raw.duplicated(["GROUP_ID", "TEAM_ID"]).any():
        raise ValueError("Expected one row per lineup id and team.")
    out = raw[list(LINEUP_RENAMES)].rename(columns=LINEUP_RENAMES)
    return out.reset_index(drop=True)


def clean_team_ratings(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[list(TEAM_RENAMES)].rename(columns=TEAM_RENAMES).reset_index(drop=True)


def _nba_date(day: date | None) -> str:
    return "" if day is None else day.strftime("%m/%d/%Y")


def load_lineups(
    client: NbaClient,
    season: str,
    group_quantity: int = 5,
    date_from: date | None = None,
    date_to: date | None = None,
    max_age: timedelta | None = None,
) -> pd.DataFrame:
    """Top ~2000 lineups of ``group_quantity`` players (by minutes) in the date range."""
    (raw,) = client.fetch(
        LeagueDashLineups,
        season=season,
        season_type_all_star="Regular Season",
        group_quantity=group_quantity,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
        date_from_nullable=_nba_date(date_from),
        date_to_nullable=_nba_date(date_to),
        max_age=max_age,
    )
    return clean_lineups(raw)


def load_team_ratings(
    client: NbaClient,
    season: str,
    date_from: date | None = None,
    date_to: date | None = None,
    max_age: timedelta | None = None,
) -> pd.DataFrame:
    """Team net rating (points per 100 possessions) over the date range."""
    (raw,) = client.fetch(
        LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
        date_from_nullable=_nba_date(date_from),
        date_to_nullable=_nba_date(date_to),
        max_age=max_age,
    )
    return clean_team_ratings(raw)
