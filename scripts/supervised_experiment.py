"""Module 1 experiment 2: logistic regression and XGBoost vs. the Elo baseline.

Train 2016-17..2022-23 | validation 2023-24, 2024-25 | final test 2025-26 (scored once).
Elo parameters come from scripts/elo_experiment.py (tuned on validation seasons only).
Run: ``python -m uv run python scripts/supervised_experiment.py``
"""

import pandas as pd

from sportslab.data import NbaClient
from sportslab.data.games import load_games, season_labels
from sportslab.features.team_form import build_pregame_features
from sportslab.models.elo import EloParams
from sportslab.models.supervised import run_supervised_experiment

TUNED_ELO = EloParams(k=24, home_advantage=40, carryover=0.7)
TRAIN = season_labels(2016, 2022)
VALIDATION = ["2023-24", "2024-25"]
TEST = "2025-26"

pd.options.display.float_format = "{:.4f}".format

games = load_games(NbaClient(), season_labels(2015, 2025))
features = build_pregame_features(games, TUNED_ELO)
exp = run_supervised_experiment(features, TRAIN, VALIDATION, TEST)

print(f"XGBoost trees chosen by early stopping on {TRAIN[-1]}: {exp.best_iteration + 1}")
print("\nVALIDATION (fit on train):")
print(exp.scores_validation.to_string())
print("\nGain in log loss vs Elo, per-game paired, 95% interval (positive = better than Elo):")
print(exp.gain_vs_elo_validation.to_string())

print(f"\nTEST {TEST} (fit on train + validation), {len(exp.test_predictions)} games:")
print(exp.scores_test.to_string())
print("\nGain in log loss vs Elo, 95% interval:")
print(exp.gain_vs_elo_test.to_string())

print("\nLogistic (all) standardized coefficients:")
print(exp.logistic_coefficients.to_string())
print("\nXGBoost feature importance (gain, normalized):")
print(exp.xgb_importance.to_string())
