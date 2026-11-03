"""Module 8: who is an All-Star / All-NBA, predicted from what was known at the time.

All-Star panel: stats through the date by which half of the season's games were played.
All-NBA panel: full regular season.
Expanding-window validation (train on earlier seasons only), first test season 2019-20.
Needs ``scripts/download_awards.py`` to have been run (everything is read from the cache).
Run: ``python -m uv run python scripts/awards_experiment.py``
"""

import pandas as pd

from sportslab.config import PROCESSED_DIR
from sportslab.data import NbaClient
from sportslab.data.awards import award_labels, load_awards_for
from sportslab.data.games import load_games, season_labels
from sportslab.data.player_season import cutoff_at_fraction, load_player_season
from sportslab.features.awards import (
    ALL_NBA_FEATURES,
    ALL_STAR_FEATURES,
    BASE_FEATURES,
    build_panel,
)
from sportslab.models.awards_model import (
    evaluate,
    expanding_window,
    make_logistic,
    make_xgb,
    winner_rank,
)

SEASONS = season_labels(2015, 2025)
FIRST_TEST = "2019-20"
MIN_GP = {"mid": 15, "end": 30}

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 220

client = NbaClient()
games = load_games(client, SEASONS)
stats = {
    kind: pd.concat(
        [
            load_player_season(client, s, cutoff_at_fraction(games, s) if kind == "mid" else None)
            for s in SEASONS
        ],
        ignore_index=True,
    )
    for kind in ("mid", "end")
}
# Same candidate lists as scripts/download_awards.py (top 80 by points per game, per panel)
top_ids = sorted(
    {
        pid
        for kind, df in stats.items()
        for s in SEASONS
        for pid in df[(df["season"] == s) & (df["gp"] >= MIN_GP[kind])].nlargest(80, "pts")[
            "player_id"
        ]
    }
)
awards = load_awards_for(client, top_ids)
labels = award_labels(awards)
print(f"{len(top_ids)} candidates, {len(awards)} award rows")

panels = {
    kind: {s: build_panel(stats[kind], awards, labels, s, 80, MIN_GP[kind]) for s in SEASONS}
    for kind in ("mid", "end")
}

print("\n=== ALL-STAR (stats through half of the schedule) ===")
models = {
    "pts_only": (make_logistic, ["pts"]),
    "stats": (make_logistic, BASE_FEATURES),
    "logit_all": (make_logistic, ALL_STAR_FEATURES),
    "xgb_all": (make_xgb, ALL_STAR_FEATURES),
}
pred = expanding_window(panels["mid"], ALL_STAR_FEATURES, "all_star", FIRST_TEST, models)
per_season, pooled = evaluate(pred, "all_star", list(models), ["pts", "pie"])
print(per_season.to_string(index=False))
print(
    "\nMean R-precision:", per_season.drop(columns=["season", "winners"]).mean().round(3).to_dict()
)
print(pooled.to_string())

print("\n=== ALL-NBA (full-season stats) ===")
models_nba = {
    "pts_only": (make_logistic, ["pts"]),
    "stats": (make_logistic, BASE_FEATURES),
    "logit_all": (make_logistic, ALL_NBA_FEATURES),
    "xgb_all": (make_xgb, ALL_NBA_FEATURES),
}
pred_nba = expanding_window(panels["end"], ALL_NBA_FEATURES, "all_nba", FIRST_TEST, models_nba)
per_nba, pooled_nba = evaluate(pred_nba, "all_nba", list(models_nba), ["pts", "pie"])
print(per_nba.to_string(index=False))
print("\nMean R-precision:", per_nba.drop(columns=["season", "winners"]).mean().round(3).to_dict())
print(pooled_nba.to_string())

print("\n=== MVP: rank of the real winner among candidates (1 = best) ===")
rows = []
for season in SEASONS:
    row = {"season": season}
    for kind in ("mid", "end"):
        panel = panels[kind][season]
        if not panel["mvp"].any():
            continue
        truth = panel["mvp"].to_numpy()
        for col in ("pie", "pts", "team_w_pct"):
            row[f"{kind}_rank_{col}"] = winner_rank(panel[col].to_numpy(), truth)
    mask = pred_nba["test_season"] == season
    if mask.any() and pred_nba.loc[mask, "mvp"].any():
        g = pred_nba[mask]
        row["end_rank_model"] = winner_rank(g["p_logit_all"].to_numpy(), g["mvp"].to_numpy())
    rows.append(row)
mvp = pd.DataFrame(rows)
print(mvp.to_string(index=False))
print(
    "\nMVP in top 3 by PIE at mid-season:",
    int((mvp["mid_rank_pie"] <= 3).sum()),
    "of",
    int(mvp["mid_rank_pie"].notna().sum()),
)
print(
    "MVP in top 3 by PIE at season end:",
    int((mvp["end_rank_pie"] <= 3).sum()),
    "of",
    int(mvp["end_rank_pie"].notna().sum()),
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
keep = [
    "test_season",
    "player_id",
    "player",
    "team",
    "pts",
    "pie",
    "team_w_pct",
    "all_star",
    "all_nba",
    "all_nba_first",
    "mvp",
]
frames = []
for name, frame in (("all_star", pred), ("all_nba", pred_nba)):
    cols = keep + [c for c in frame.columns if c.startswith("p_")]
    frames.append(frame[cols].assign(panel=name))
pd.concat(frames, ignore_index=True).to_parquet(PROCESSED_DIR / "awards_predictions.parquet")
pd.concat(
    [per_season.assign(panel="all_star"), per_nba.assign(panel="all_nba")], ignore_index=True
).to_parquet(PROCESSED_DIR / "awards_per_season.parquet")
pd.concat(
    [
        pooled.assign(panel="all_star").reset_index(names="model"),
        pooled_nba.assign(panel="all_nba").reset_index(names="model"),
    ],
    ignore_index=True,
).to_parquet(PROCESSED_DIR / "awards_pooled.parquet")
mvp.to_parquet(PROCESSED_DIR / "awards_mvp.parquet")
print("saved results to", PROCESSED_DIR)
