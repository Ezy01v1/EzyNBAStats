"""Module 2 hypothesis B on real data: points per shot by court zone, league-wide.

The league zone totals come with every ShotChartDetail response and are identical for every
player in a season, so one request per season is enough.
Run: ``python -m uv run python scripts/shot_hypotheses.py``
"""

import pandas as pd

from sportslab.analysis.shots import BASIC, BASIC_RANGE, league_zone_table, zone_table
from sportslab.data import NbaClient
from sportslab.data.games import season_labels
from sportslab.data.shots import load_player_shots

REFERENCE_PLAYER = 203999  # any player who shot that season works; the league frame is the same
SEASON = "2025-26"
STABILITY = season_labels(2021, 2025)

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 160

client = NbaClient()
leagues = {s: load_player_shots(client, REFERENCE_PLAYER, s)[1] for s in STABILITY}

print(f"=== B. League points per shot by zone, {SEASON} ===")
print(league_zone_table(leagues[SEASON], BASIC).sort_values("pps", ascending=False).to_string())
print("\nBy zone and distance range (pps):")
print(
    league_zone_table(leagues[SEASON], BASIC_RANGE).sort_values("pps", ascending=False).to_string()
)

print("\nPoints per shot by zone, every season (rank 1 = best):")
by_season = pd.DataFrame(
    {s: league_zone_table(lg, BASIC)["pps"] for s, lg in leagues.items()}
).sort_values(SEASON, ascending=False)
print(by_season.to_string())

print("\nLong mid-range (16-24 ft) vs three-point zones, pps by season:")
rows = {}
for s, lg in leagues.items():
    fine = league_zone_table(lg, BASIC_RANGE)["pps"]
    basic = league_zone_table(lg, BASIC)["pps"]
    rows[s] = {
        "mid 16-24": fine[("Mid-Range", "16-24 ft.")],
        "above-break 3": basic["Above the Break 3"],
        "corner 3 (L)": basic["Left Corner 3"],
        "corner 3 (R)": basic["Right Corner 3"],
    }
print(pd.DataFrame(rows).T.to_string())

print("\nSanity: the player-level table for the reference player (zone_table):")
shots, league = load_player_shots(client, REFERENCE_PLAYER, SEASON)
print(
    zone_table(shots, league)[
        [
            "fga",
            "fg_pct",
            "ci_low",
            "ci_high",
            "league_fg_pct",
            "pps",
            "league_pps",
            "excludes_league",
        ]
    ].to_string()
)
