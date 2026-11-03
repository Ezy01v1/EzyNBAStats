"""Download what Module 8 needs and report how well the candidate lists cover the awards.

For each season 2015-16..2025-26 the candidates are the 80 players with the most points per game
halfway through the schedule (All-Star panel) and at the end of the regular season (All-NBA panel).
One PlayerAwards request per candidate (~1 s each the first time, cached afterwards).
Run: ``python -m uv run python scripts/download_awards.py``
"""

import pandas as pd

from sportslab.data import NbaClient
from sportslab.data.awards import award_labels, load_awards_for
from sportslab.data.games import load_games, season_labels
from sportslab.data.player_season import cutoff_at_fraction, load_player_season

SEASONS = season_labels(2015, 2025)
N_CANDIDATES = 80
MIN_GP = {"mid": 15, "end": 30}

pd.options.display.width = 200
client = NbaClient()
games = load_games(client, SEASONS)

panels = {}
for season in SEASONS:
    for kind, day in (("mid", cutoff_at_fraction(games, season)), ("end", None)):
        stats = load_player_season(client, season, day)
        pool = stats[stats["gp"] >= MIN_GP[kind]].nlargest(N_CANDIDATES, "pts")
        panels[(season, kind)] = pool
ids = sorted({pid for pool in panels.values() for pid in pool["player_id"]})
print(
    f"{len(ids)} distinct candidates: downloading awards (cached ones are instant)...", flush=True
)
awards = load_awards_for(client, ids)
labels = award_labels(awards)

rows = []
for (season, kind), pool in panels.items():
    lab = labels[labels["season"] == season]
    in_pool = lab[lab["player_id"].isin(pool["player_id"])]
    rows.append(
        {
            "season": season,
            "panel": kind,
            "all_star_in_pool": int(in_pool["all_star"].sum()),
            "all_star_total_seen": int(lab["all_star"].sum()),
            "all_nba_in_pool": int(in_pool["all_nba"].sum()),
            "mvp_in_pool": int(in_pool["mvp"].sum()),
        }
    )
cov = pd.DataFrame(rows)
print(cov.to_string(index=False))
