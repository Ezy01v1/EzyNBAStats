"""Player-season advanced stats from NBA.com (LeagueDashPlayerStats, measure type Advanced).

PER and BPM are not published by NBA.com (they come from Basketball Reference); PIE is the
closest NBA.com offers. NET_RATING is a *team* quantity measured while the player is on court.
"""

from collections.abc import Sequence
from datetime import timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats

from sportslab.data.client import NbaClient

RENAMES = {
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player",
    "TEAM_ABBREVIATION": "team",
    "AGE": "age",
    "GP": "gp",
    "MIN": "min_per_game",
    "TS_PCT": "ts_pct",
    "USG_PCT": "usg_pct",
    "AST_PCT": "ast_pct",
    "REB_PCT": "reb_pct",
    "OFF_RATING": "off_rating",
    "DEF_RATING": "def_rating",
    "NET_RATING": "net_rating",
    "PIE": "pie",
    "POSS": "poss",
}
PLAYER_COLUMNS = ["season", *RENAMES.values(), "minutes"]

# A team plays ~100 possessions per 48 minutes (~2.1 per minute); a player on court sees the same.
POSS_PER_MINUTE_RANGE = (1.5, 3.0)


def clean_player_stats(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Select and rename the columns we use, and derive total minutes played.

    With ``per_mode_detailed="Totals"`` the Advanced endpoint still returns MIN as a per-game
    average (verified against real data), so total minutes = MIN * GP. The possessions-per-minute
    check fails loudly if NBA.com ever changes that.
    """
    if raw["PLAYER_ID"].duplicated().any():
        raise ValueError("Expected one row per player; traded players should come pre-aggregated.")
    out = raw[list(RENAMES)].rename(columns=RENAMES)
    out.insert(0, "season", season)
    out["minutes"] = out["min_per_game"] * out["gp"]
    played = out[out["minutes"] > 0]
    if not played.empty:
        poss_per_min = (played["poss"] / played["minutes"]).median()
        low, high = POSS_PER_MINUTE_RANGE
        if not low <= poss_per_min <= high:
            raise ValueError(
                f"Median possessions per minute is {poss_per_min:.2f}, outside {low}-{high}: "
                "MIN may not be per-game any more. Check the endpoint before trusting minutes."
            )
    return out[PLAYER_COLUMNS].reset_index(drop=True)


def load_player_stats(
    client: NbaClient, seasons: Sequence[str], max_age: timedelta | None = None
) -> pd.DataFrame:
    """Regular-season advanced stats, one row per player per season (all players, any minutes)."""
    frames = []
    for season in seasons:
        (raw,) = client.fetch(
            LeagueDashPlayerStats,
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Totals",
            max_age=max_age,
        )
        frames.append(clean_player_stats(raw, season))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PLAYER_COLUMNS)
