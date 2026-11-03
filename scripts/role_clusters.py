"""Module 4 on real data: do rotation players fall into natural roles?

Run: ``python -m uv run python scripts/role_clusters.py``
"""

import pandas as pd

from sportslab.analysis.roles import (
    cluster_profile,
    cross_season_ari,
    fit_kmeans,
    k_diagnostics,
    nearest_players,
    relabel_by_size,
)
from sportslab.data import NbaClient
from sportslab.data.players import load_player_stats
from sportslab.data.profiles import load_per100, load_shot_zones
from sportslab.features.roles import FEATURES, build_role_features, standardize

SEASON, PREVIOUS = "2025-26", "2024-25"
KS = range(2, 11)

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 200

client = NbaClient()


def season_features(season: str) -> pd.DataFrame:
    return build_role_features(
        load_player_stats(client, [season]),
        load_per100(client, season),
        load_shot_zones(client, season),
    )


now, before = season_features(SEASON), season_features(PREVIOUS)
z, _, _ = standardize(now)
z_prev, _, _ = standardize(before)
print(f"{SEASON}: {len(now)} players, {len(FEATURES)} features; {PREVIOUS}: {len(before)} players")

diag = k_diagnostics(z, KS, n_boot=100, n_null=20)
diag["cross_season_ari"] = [cross_season_ari(z_prev, z, k) for k in KS]
print(diag.to_string(index=False))

for k in (4, 5, 6):
    model = fit_kmeans(z, k)
    labels = relabel_by_size(model.labels_)
    prof = cluster_profile(z, labels)
    print(f"\n=== k={k}: cluster profile (mean z-score; size first) ===")
    print(prof.round(2).to_string())
    for c in range(k):
        members = now.loc[labels == c].nlargest(6, "minutes")["player"].tolist()
        print(f"  cluster {c} ({int((labels == c).sum())}): {', '.join(members)}")

print("\nNearest players to Nikola Jokić:")
print(nearest_players(z, now["player"], "Nikola Jokić").to_string(index=False))
