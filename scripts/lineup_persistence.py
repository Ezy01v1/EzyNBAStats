"""Module 7 on real data: do lineup net ratings persist from the first to the second half?

5-man lineups barely overlap between halves (a handful with 100+ possessions in both), so the
test runs on 2-man and 3-man groups, which have enough data. Run:
``python -m uv run python scripts/lineup_persistence.py``
"""

import pandas as pd

from sportslab.analysis.lineups import (
    add_uncertainty,
    margin_sd_per_game,
    paired_error_gain,
    paired_halves,
    persistence,
    prediction_errors,
)
from sportslab.data import NbaClient
from sportslab.data.games import load_games
from sportslab.data.lineups import load_lineups, load_team_ratings

SEASON = "2025-26"
MIN_POSS = 300
KS = (0, 100, 200, 400, 800, 1600, 3200, 1_000_000)

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 200

client = NbaClient()
games = load_games(client, [SEASON])
dates = sorted(games["game_date"].dt.date.unique())
mid = len(dates) // 2
first_to, second_from = dates[mid - 1], dates[mid]
margin_sd = margin_sd_per_game(games)
print(f"{SEASON}: halves split at {second_from}; per-game margin SD {margin_sd:.1f}")

team_first = load_team_ratings(client, SEASON, date_to=first_to)

# 5-man lineups: coverage and sampling error over the whole season
five = load_lineups(client, SEASON, 5)
team_minutes = 30 * 82 * 48
print(
    f"\n5-man lineups: {len(five)} returned, smallest has {five.minutes.min():.0f} min, "
    f"they cover {five.minutes.sum() / team_minutes:.0%} of all lineup minutes"
)
big = add_uncertainty(five[five["minutes"] >= 100], margin_sd)
print(f"with >= 100 min: {len(big)} lineups; median SE {big.se.median():.1f} net-rating points")
print(
    f"intervals that exclude 0: {int(((big.ci_low > 0) | (big.ci_high < 0)).sum())} of {len(big)}"
)

for size in (2, 3, 5):
    first = load_lineups(client, SEASON, size, date_to=first_to)
    second = load_lineups(client, SEASON, size, date_from=second_from)
    joined = paired_halves(first, second, team_first, MIN_POSS)
    print(f"\n=== groups of {size}: {len(joined)} in both halves, >= {MIN_POSS} poss each ===")
    if len(joined) < 30:
        print("too few for a meaningful test")
        continue
    p = persistence(joined)
    print(
        "Spearman first vs second half:",
        " ".join(f"{p['raw'][i]:+.3f}" for i in range(3)),
        "(rho, low, high)",
    )
    print(
        "   beyond the team           :", " ".join(f"{p['beyond_team'][i]:+.3f}" for i in range(3))
    )
    for prior in ("team", "zero"):
        err = prediction_errors(joined, KS, prior)
        print(f"weighted MSE, prior={prior}: ", {k: round(v, 1) for k, v in err.items()})
        print(f"   best k = {int(err.idxmin())}; raw (k=0) {err[0]:.1f} -> best {err.min():.1f}")
        if prior == "team":
            g, lo, hi = paired_error_gain(joined, 0, int(err.idxmin()), prior)
            print(f"   gain of best k over raw: {g:.1f} [{lo:.1f}, {hi:.1f}]")
            g, lo, hi = paired_error_gain(joined, 1_000_000, int(err.idxmin()), prior)
            print(f"   gain of best k over the team rating alone: {g:.1f} [{lo:.1f}, {hi:.1f}]")
