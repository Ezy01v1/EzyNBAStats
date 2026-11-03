"""Module 3 hypotheses A and C on real data (usage vs. efficiency; net rating vs. PIE).

Run: ``python -m uv run python scripts/player_hypotheses.py``
"""

import pandas as pd

from sportslab.analysis.comparator import (
    binned_means,
    eligible_pool,
    percentile_profile,
    rank_correlation,
    top_overlap,
)
from sportslab.data import NbaClient
from sportslab.data.games import season_labels
from sportslab.data.players import load_player_stats

SEASON = "2025-26"
STABILITY = season_labels(2021, 2025)

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 160

players = load_player_stats(NbaClient(), STABILITY)
pool = eligible_pool(players[players["season"] == SEASON])
print(f"{SEASON}: {len(pool)} players with >= 1000 minutes (of {(players.season == SEASON).sum()})")

print("\n=== A. USG% vs TS% (Spearman, 95% bootstrap interval) ===")
rho, lo, hi = rank_correlation(pool["usg_pct"], pool["ts_pct"])
print(f"{SEASON} raw:                  {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]")
rho, lo, hi = rank_correlation(pool["usg_pct"], pool["ts_pct"], control=pool["reb_pct"])
print(f"{SEASON} controlling REB%:     {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("\nTS% by USG% quintile:")
print(binned_means(pool, "usg_pct", "ts_pct").to_string())
print("\nStability by season (raw / controlling REB%):")
for season in STABILITY:
    p = eligible_pool(players[players["season"] == season])
    raw = rank_correlation(p["usg_pct"], p["ts_pct"], n_boot=500)
    ctl = rank_correlation(p["usg_pct"], p["ts_pct"], control=p["reb_pct"], n_boot=500)
    print(
        f"{season} n={len(p):3d}  raw {raw[0]:+.3f} [{raw[1]:+.3f}, {raw[2]:+.3f}]"
        f"   ctl {ctl[0]:+.3f} [{ctl[1]:+.3f}, {ctl[2]:+.3f}]"
    )

print("\n=== C. NET_RATING vs PIE ===")
rho, lo, hi = rank_correlation(pool["net_rating"], pool["pie"])
print(f"{SEASON} Spearman: {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]")
n_shared, shared = top_overlap(pool, "net_rating", "pie", n=20)
print(f"Top-20 overlap: {n_shared}/20 -> {sorted(shared)}")
pct = percentile_profile(pool, pool, ["net_rating", "pie"])
gap = (pct["net_rating"] - pct["pie"]).rename("gap")
view = (
    pool[["player", "team", "minutes", "net_rating", "pie"]].join(pct.add_suffix("_pct")).join(gap)
)
print("\nBiggest 'net rating >> PIE' (teammates carry them?):")
print(view.nlargest(6, "gap").to_string(index=False))
print("\nBiggest 'PIE >> net rating' (good player, bad team?):")
print(view.nsmallest(6, "gap").to_string(index=False))

print("\n=== Top 8 by PIE: percentile profile ===")
top = pool.nlargest(8, "pie")
prof = percentile_profile(top, pool).round(0)
print(top[["player", "team", "minutes"]].join(prof).to_string(index=False))
