"""Does shrinking zone shooting percentages towards the league rate predict better?

For the 100 players with most minutes in 2025-26, use the first half of their game dates to
predict zone field-goal rates in the second half (chronological split, never shuffled), with
different amounts of shrinkage. ~100 ShotChartDetail requests the first time, then cached.
Run: ``python -m uv run python scripts/shot_shrinkage.py``
"""

import pandas as pd

from sportslab.analysis.shots import BASIC, half_season_counts, league_zone_table, shrinkage_error
from sportslab.data import NbaClient
from sportslab.data.players import load_player_stats
from sportslab.data.shots import load_player_shots

SEASON = "2025-26"
N_PLAYERS = 100
PRIORS = (0, 5, 10, 20, 40, 80, 160, 320, 1_000_000)  # last one ~ "always the league rate"

pd.options.display.float_format = "{:.5f}".format
pd.options.display.width = 160

client = NbaClient()
players = load_player_stats(client, [SEASON]).nlargest(N_PLAYERS, "minutes")

frames, league = [], None
for player_id in players["player_id"]:
    shots, league = load_player_shots(client, int(player_id), SEASON)
    counts = half_season_counts(shots, BASIC).reset_index()
    counts["player_id"] = player_id
    frames.append(counts)
counts = pd.concat(frames, ignore_index=True)
counts["league_rate"] = counts["zone_basic"].map(league_zone_table(league, BASIC)["fg_pct"])
print(
    f"{len(players)} players, {len(counts)} player-zones, "
    f"{int(counts.fga_1.sum())} first-half shots"
)

rows = {}
for zone, part in [("ALL", counts), *counts.groupby("zone_basic")]:
    rows[zone] = {k: shrinkage_error(part, k) for k in PRIORS}
table = pd.DataFrame(rows).T
table.columns = [f"k={k}" if k < 1_000_000 else "league only" for k in PRIORS]
print("\nWeighted MSE of the second-half rate (lower is better):")
print(table.to_string())
print("\nBest k per zone:", table.idxmin(axis=1).to_dict())
raw, best = table.loc["ALL", "k=0"], table.loc["ALL"].min()
print(f"\nAll zones: raw {raw:.5f} -> best {best:.5f} ({1 - best / raw:.1%} lower error)")
