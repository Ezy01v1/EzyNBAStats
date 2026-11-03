"""Player-season stats as known on a given date (or at the end of the season).

Award voting happens mid-season, so a model of it must use what was known then: stats through a
cutoff date (halfway through the schedule). Using the full season would leak the future into the
prediction.
"""

from datetime import date

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats

from sportslab.data.client import NbaClient

PLAYER_COLUMNS = [
    "season",
    "player_id",
    "player",
    "team_id",
    "team",
    "age",
    "gp",
    "min_pg",
    "pts",
    "reb",
    "ast",
    "ts_pct",
    "usg_pct",
    "pie",
    "team_w_pct",
]


def cutoff_at_fraction(games: pd.DataFrame, season: str, fraction: float = 0.5) -> date:
    """Date by which ``fraction`` of the season's games had been played.

    A fixed calendar date does not work: in the shortened 2020-21 season only ~12 games per team
    had been played by January 15, so halfway through the schedule is the comparable moment.
    ``games`` needs ``season`` and ``game_date`` (as ``load_games`` returns).
    """
    dates = games.loc[games["season"] == season, "game_date"].sort_values().reset_index(drop=True)
    if dates.empty:
        raise ValueError(f"no games for {season}")
    index = min(int(fraction * len(dates)), len(dates) - 1)
    return dates.iloc[index].date()


def _nba_date(day: date | None) -> str:
    return "" if day is None else day.strftime("%m/%d/%Y")


def build_player_season(
    base: pd.DataFrame, advanced: pd.DataFrame, teams: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Join per-game box score, advanced rates and the team's win percentage."""
    adv = advanced[["PLAYER_ID", "TS_PCT", "USG_PCT", "PIE"]]
    out = base.merge(adv, on="PLAYER_ID", validate="one_to_one")
    out = out.merge(
        teams[["TEAM_ID", "W_PCT"]].rename(columns={"W_PCT": "team_w_pct"}),
        on="TEAM_ID",
        how="left",
        validate="many_to_one",
    )
    out = out.rename(
        columns={
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team",
            "AGE": "age",
            "GP": "gp",
            "MIN": "min_pg",
            "PTS": "pts",
            "REB": "reb",
            "AST": "ast",
            "TS_PCT": "ts_pct",
            "USG_PCT": "usg_pct",
            "PIE": "pie",
        }
    )
    out.insert(0, "season", season)
    return out[PLAYER_COLUMNS].reset_index(drop=True)


def load_player_season(client: NbaClient, season: str, date_to: date | None = None) -> pd.DataFrame:
    """Per-game stats of every player through ``date_to`` (None = end of the regular season)."""
    common = {
        "season": season,
        "season_type_all_star": "Regular Season",
        "date_to_nullable": _nba_date(date_to),
    }
    (base,) = client.fetch(
        LeagueDashPlayerStats,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
        **common,
    )
    (adv,) = client.fetch(
        LeagueDashPlayerStats,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
        **common,
    )
    (teams,) = client.fetch(
        LeagueDashTeamStats,
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        **common,
    )
    return build_player_season(base, adv, teams, season)
