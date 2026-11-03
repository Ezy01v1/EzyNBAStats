"""Module 6 backtest: how good are the simulator's statements at 10/30/50/70% of the season?

Seasons 2021-22..2025-26 (the play-in exists from 2020-21; 2020-21 is skipped for its COVID
schedule). Elo parameters are the ones tuned in Module 1, which used 2023-24 and 2024-25 as
validation: those two seasons are NOT a clean test, 2025-26 was seen many times too.

A first run with frozen ratings (rating_sd = 0) gave intervals far too narrow, so the grid below
adds uncertainty about each team's true strength and picks it by leave-one-season-out: the sd
for a held-out season is chosen on the other four.
Run: ``python -m uv run python scripts/season_backtest.py``
"""

import numpy as np
import pandas as pd

from sportslab.config import PROCESSED_DIR
from sportslab.data import NbaClient
from sportslab.data.games import load_games, season_labels
from sportslab.data.schedule import load_conferences, load_playoff_games, playoff_outcomes
from sportslab.models.elo import EloParams
from sportslab.models.season_backtest import backtest_season, summarize_backtest

TUNED_ELO = EloParams(k=24, home_advantage=40, carryover=0.7)
FRACTIONS = (0.1, 0.3, 0.5, 0.7)
RATING_SDS = (0, 25, 50, 75, 100, 150)
N_SIMS = 4000
SEASONS = season_labels(2021, 2025)
NOMINAL = {"50%": 0.5, "80%": 0.8, "90%": 0.9}

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 220

client = NbaClient()
games = load_games(client, season_labels(2015, 2025))

frames = []
for i, season in enumerate(SEASONS):
    conferences = load_conferences(client, season)
    outcome = playoff_outcomes(load_playoff_games(client, season))
    upto = games[games["season"] <= season]
    for sd in RATING_SDS:
        frames.append(
            backtest_season(
                upto,
                season,
                FRACTIONS,
                conferences,
                outcome,
                TUNED_ELO,
                n_sims=N_SIMS,
                seed=100 * i,
                rating_sd=sd,
            )
        )
    print(f"{season} done", flush=True)
frame = pd.concat(frames, ignore_index=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
frame.to_parquet(PROCESSED_DIR / "season_backtest_grid.parquet", index=False)

summary = summarize_backtest(frame, N_SIMS)
summary["calib_gap"] = sum(
    (summary[f"coverage_{name}"] - nominal).abs() for name, nominal in NOMINAL.items()
) / len(NOMINAL)

print("\nAverage over seasons and cutoffs, by rating_sd:")
by_sd = summary.groupby("rating_sd").agg(
    cov50=("coverage_50%", "mean"),
    cov80=("coverage_80%", "mean"),
    cov90=("coverage_90%", "mean"),
    gap=("calib_gap", "mean"),
    mae=("mae_wins_sim", "mean"),
    brier=("brier_playoffs_sim", "mean"),
    logscore=("logscore_title", "mean"),
    top_p=("top_title_prob", "mean"),
)
print(by_sd.to_string())

print("\nCoverage of the 80% interval by rating_sd and cutoff (nominal 0.80):")
print(summary.pivot_table(index="rating_sd", columns="cutoff", values="coverage_80%").to_string())

print("\nLeave-one-season-out: sd chosen on the other 4 seasons, scored on the held-out one")
rows = []
for held in SEASONS:
    train = summary[summary["season"] != held].groupby("rating_sd")["calib_gap"].mean()
    best = float(train.idxmin())
    test = summary[(summary["season"] == held) & (summary["rating_sd"] == best)]
    zero = summary[(summary["season"] == held) & (summary["rating_sd"] == 0)]
    rows.append(
        {
            "held_out": held,
            "chosen_sd": best,
            "cov80_chosen": test["coverage_80%"].mean(),
            "cov80_frozen": zero["coverage_80%"].mean(),
            "logscore_chosen": test["logscore_title"].mean(),
            "logscore_frozen": zero["logscore_title"].mean(),
            "brier_chosen": test["brier_playoffs_sim"].mean(),
            "brier_frozen": zero["brier_playoffs_sim"].mean(),
        }
    )
loso = pd.DataFrame(rows)
print(loso.to_string(index=False))
print("\nMeans:", loso.drop(columns=["held_out"]).mean().round(3).to_dict())
print(f"Uniform title log score: {np.log(30):.3f}")
