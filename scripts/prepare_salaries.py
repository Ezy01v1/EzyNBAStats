"""Convert the Kaggle file omarsobhy14/nba-players-salaries into our salaries.csv schema.

The raw file is wide (one column per season, "$48,070,014" strings, $0 = no contract known) and
its LAST column is mislabeled: it is headed "2024/2025.1" (a duplicate), but its values are the
2025-26 salaries (e.g. Stephen Curry $59,606,817, Kevin Durant $54,708,608). The columns after
2022-23 look like contract schedules known when the data was collected, so they cover only
players already under contract: run the coverage check before trusting any season.

Usage: ``python -m uv run python scripts/prepare_salaries.py``
"""

import pandas as pd

from sportslab.config import DATA_DIR
from sportslab.data import NbaClient
from sportslab.data.players import load_player_stats
from sportslab.data.salaries import match_salaries, read_salaries

RAW = DATA_DIR / "external" / "kaggle_omarsobhy14_raw.csv"
OUT = DATA_DIR / "external" / "salaries.csv"
SEASON_BY_COLUMN = {
    "2022/2023": "2022-23",
    "2023/2024": "2023-24",
    "2024/2025": "2024-25",
    "2024/2025.1": "2025-26",  # mislabeled duplicate header, see module docstring
}
MIN_MINUTES = 1000

pd.options.display.width = 200

raw = pd.read_csv(RAW, dtype=str)
long = raw.melt(
    id_vars=["Player Name"],
    value_vars=list(SEASON_BY_COLUMN),
    var_name="column",
    value_name="salary",
)
long["season"] = long["column"].map(SEASON_BY_COLUMN)
long["salary_usd"] = long["salary"].str.replace(r"[$,\s]", "", regex=True).astype(float)
long = long[long["salary_usd"] > 0]  # $0 = no contract known for that season
long = long.rename(columns={"Player Name": "player_name"})[["season", "player_name", "salary_usd"]]
long["salary_usd"] = long["salary_usd"].astype(int)
long.sort_values(["season", "salary_usd"], ascending=[True, False]).to_csv(OUT, index=False)
per_season = long["season"].value_counts().sort_index().to_dict()
print(f"wrote {OUT.name}: {len(long)} rows; rows per season: {per_season}")

salaries = read_salaries(OUT)
client = NbaClient()
rows = []
for season in SEASON_BY_COLUMN.values():
    players = load_player_stats(client, [season])
    matched, csv_left, players_left = match_salaries(
        salaries[salaries["season"] == season], players
    )
    rotation = players[players["minutes"] >= MIN_MINUTES]
    covered = matched[matched["minutes"] >= MIN_MINUTES]
    rows.append(
        {
            "season": season,
            "csv_rows": int((salaries["season"] == season).sum()),
            "matched": len(matched),
            "csv_unmatched": len(csv_left),
            "rotation_players": len(rotation),
            "rotation_with_salary": len(covered),
            "coverage": len(covered) / len(rotation),
        }
    )
print(pd.DataFrame(rows).to_string(index=False, float_format="{:.2f}".format))
