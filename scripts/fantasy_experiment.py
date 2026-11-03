"""Module 10: project the fantasy points of a player's next game, with intervals.

Protocol (temporal, never shuffled), as in Module 1:
  Stage A: fit on 2022-23, calibrate intervals on 2023-24, score validation 2024-25.
  Stage B: fit on 2022-23 + 2023-24, calibrate intervals on 2024-25, score test 2025-26 ONCE.
2021-22 only supplies "last season's average" for 2022-23. Rows are "if he plays".
Run: ``python -m uv run python scripts/fantasy_experiment.py``
"""

import numpy as np
import pandas as pd

from sportslab.config import PROCESSED_DIR
from sportslab.data import NbaClient
from sportslab.data.gamelog import load_player_games
from sportslab.features.fantasy import build_features, eligible_rows, fill_missing
from sportslab.models.fantasy_model import (
    IntervalCalibrator,
    baseline_last5,
    baseline_season,
    coverage,
    fit_ridge,
    fit_shrinkage,
    paired_abs_error_gain,
    predict_ridge,
    shrunk_mean,
)

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
KS = (0, 3, 5, 10, 20, 40, 80)

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 200

games = load_player_games(NbaClient(), SEASONS)
rows = fill_missing(eligible_rows(build_features(games)))
rows = rows[rows["season"] != "2021-22"].reset_index(drop=True)
print(
    f"{len(rows):,} player-games; "
    f"per season: {rows['season'].value_counts().sort_index().to_dict()}"
)
print(f"mean fantasy points: {rows['target'].mean():.1f}, sd {rows['target'].std():.1f}")


def predictions(frame: pd.DataFrame, k: float, ridge) -> pd.DataFrame:
    out = frame[["season", "game_date", "player_id", "player", "team", "opponent", "target"]].copy()
    out["season_avg"] = baseline_season(frame).to_numpy()
    out["last5"] = baseline_last5(frame).to_numpy()
    out["ewm10"] = frame["last10_ewm"].to_numpy()
    out["shrunk"] = shrunk_mean(frame, k).to_numpy()
    out["ridge"] = predict_ridge(ridge, frame)
    return out


def stage(train_seasons, calib_season, score_season):
    train = rows[rows["season"].isin(train_seasons)]
    k = fit_shrinkage(train, KS)
    ridge = fit_ridge(train)
    calib = predictions(rows[rows["season"] == calib_season], k, ridge)
    score = predictions(rows[rows["season"] == score_season], k, ridge)
    return k, calib, score


def report(name, calib, score):
    print(f"\n=== {name} ===")
    table = {}
    for col in ("season_avg", "last5", "ewm10", "shrunk", "ridge"):
        err = score["target"] - score[col]
        table[col] = {
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err**2).mean()),
            "bias": -err.mean(),
        }
    print(pd.DataFrame(table).T.to_string())
    print("\nPaired gain in MAE (positive = better), bootstrap over players:")
    for new, ref in (
        ("last5", "season_avg"),
        ("shrunk", "season_avg"),
        ("ridge", "season_avg"),
        ("ridge", "shrunk"),
    ):
        g, lo, hi = paired_abs_error_gain(
            score["target"], score[new], score[ref], score["player_id"]
        )
        print(f"  {new} vs {ref}: {g:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(
        "\nInterval coverage (nominal 50% / 80%) and mean width "
        "(intervals calibrated on the previous season):"
    )
    for col in ("season_avg", "ridge"):
        cal = IntervalCalibrator().fit(calib[col], calib["target"])
        cells = []
        for level in (0.5, 0.8):
            low, high = cal.interval(score[col].to_numpy(), level)
            cells.append(
                f"{level:.0%}: cov {coverage(score['target'], low, high):.3f} "
                f"width {np.mean(high - low):.1f}"
            )
        print(f"  {col:<11s} " + " | ".join(cells))


k_a, calib_a, score_a = stage(["2022-23"], "2023-24", "2024-25")
print(f"\nStage A: shrinkage k chosen = {k_a}")
report("VALIDATION 2024-25 (fit 2022-23, calibrate 2023-24)", calib_a, score_a)

k_b, calib_b, score_b = stage(["2022-23", "2023-24"], "2024-25", "2025-26")
print(f"\nStage B: shrinkage k chosen = {k_b}")
report("TEST 2025-26 (fit 2022-23 + 2023-24, calibrate 2024-25)", calib_b, score_b)

# Save test-season predictions with ridge intervals for the page
cal = IntervalCalibrator().fit(calib_b["ridge"], calib_b["target"])
saved = score_b.copy()
for level in (0.5, 0.8):
    low, high = cal.interval(saved["ridge"].to_numpy(), level)
    saved[f"low{int(level * 100)}"], saved[f"high{int(level * 100)}"] = low, high
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
saved.to_parquet(PROCESSED_DIR / "fantasy_test_predictions.parquet", index=False)

# By level: where is the error?
score_b["level"] = pd.qcut(score_b["ridge"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
by_level = score_b.groupby("level", observed=True).apply(
    lambda g: pd.Series(
        {
            "n": len(g),
            "mean pred": g["ridge"].mean(),
            "MAE ridge": (g["target"] - g["ridge"]).abs().mean(),
        }
    ),
    include_groups=False,
)
print("\nTest error by predicted level (ridge):")
print(by_level.to_string())
