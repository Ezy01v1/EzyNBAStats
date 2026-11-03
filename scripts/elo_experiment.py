"""Module 1 experiment: Elo vs. naive baselines with strict temporal validation.

Warm-up: 2015-16 .. 2022-23 | tuning: 2023-24, 2024-25 | final test (looked at once): 2025-26.
Run: ``python -m uv run python scripts/elo_experiment.py``
"""

import numpy as np

from sportslab.data import NbaClient
from sportslab.data.games import load_games, season_labels
from sportslab.models.elo import EloParams, run_elo, scoring_games, tune_elo
from sportslab.models.metrics import calibration_table, score_all

VALIDATION = ["2023-24", "2024-25"]
TEST = "2025-26"

games = load_games(NbaClient(), season_labels(2015, 2025))
real = games[~games["neutral_site"]]
print(f"games: {len(games)} ({games['neutral_site'].sum()} neutral-site, excluded from home stats)")
print(f"home win rate, all seasons: {real['home_win'].mean():.3f}")
print("by season:", real.groupby("season")["home_win"].mean().round(3).to_dict())

grid = tune_elo(
    games[games["season"] != TEST],  # the test season is not even loaded into tuning
    VALIDATION,
    ks=[4, 8, 12, 16, 20, 24, 32, 40],
    home_advantages=[40, 60, 80, 100, 120],
    carryovers=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
)
print("\nTop 5 on validation:")
print(grid.head(5).round(4).to_string(index=False))
print("Worst 2:")
print(grid.tail(2).round(4).to_string(index=False))

best = grid.iloc[0]
tuned = EloParams(k=best["k"], home_advantage=best["home_advantage"], carryover=best["carryover"])

test = scoring_games(games, [TEST])
y = test["home_win"].to_numpy()
train_rate = scoring_games(games, set(games["season"]) - {TEST})["home_win"].mean()
print(f"\nTEST {TEST}: {len(y)} games, home win rate {y.mean():.3f} (train rate {train_rate:.3f})")


def elo_p(params: EloParams) -> np.ndarray:
    out = run_elo(games, params)
    return scoring_games(out, [TEST])["p_home"].to_numpy()


results = {
    "coin 50%": score_all(y, np.full(len(y), 0.5)),
    "home rate (train)": score_all(y, np.full(len(y), train_rate)),
    "elo default (k20,h100,c.75)": score_all(y, elo_p(EloParams())),
    f"elo tuned (k{best['k']:.0f},h{best['home_advantage']:.0f},c{best['carryover']})": score_all(
        y, elo_p(tuned)
    ),
}
for name, s in results.items():
    print(
        f"{name:34s} log_loss {s['log_loss']:.4f}  brier {s['brier']:.4f}  acc {s['accuracy']:.3f}"
    )

print("\nCalibration of tuned Elo on test:")
print(calibration_table(y, elo_p(tuned), n_bins=10).round(3).to_string(index=False))
