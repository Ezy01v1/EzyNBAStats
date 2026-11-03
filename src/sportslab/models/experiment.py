"""Module 1 experiment: Elo vs. naive baselines under strict temporal validation.

Ratings warm up on early seasons, parameters are chosen on ``validation`` seasons only, and the
``test`` season is scored once at the end. The tuning step never receives the test season, so
its labels cannot influence the chosen parameters. The notebook and the Streamlit page both call
this, so they always agree.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from sportslab.models.elo import EloParams, run_elo, scoring_games, tune_elo
from sportslab.models.metrics import calibration_table, score_all

DEFAULT_KS = (4, 8, 12, 16, 20, 24, 32, 40)
DEFAULT_HOME_ADVANTAGES = (40, 60, 80, 100, 120)
DEFAULT_CARRYOVERS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class EloExperiment:
    grid: pd.DataFrame  # validation log loss per parameter combo, best first
    tuned: EloParams
    scores: pd.DataFrame  # one row per model, on the test season
    test_predictions: pd.DataFrame  # test games with p_coin, p_home_rate, p_elo
    calibration: pd.DataFrame  # tuned Elo on the test season


def run_elo_experiment(
    games: pd.DataFrame,
    validation: Sequence[str],
    test: str,
    ks: Iterable[float] = DEFAULT_KS,
    home_advantages: Iterable[float] = DEFAULT_HOME_ADVANTAGES,
    carryovers: Iterable[float] = DEFAULT_CARRYOVERS,
) -> EloExperiment:
    pre_test = games[games["season"] < test]  # "YYYY-YY" labels sort chronologically
    grid = tune_elo(pre_test, validation, ks, home_advantages, carryovers)
    best = grid.iloc[0]
    tuned = EloParams(
        k=float(best["k"]),
        home_advantage=float(best["home_advantage"]),
        carryover=float(best["carryover"]),
    )

    train_rate = float(scoring_games(pre_test, set(pre_test["season"]))["home_win"].mean())
    default_p = scoring_games(run_elo(games, EloParams()), [test])
    tuned_p = scoring_games(run_elo(games, tuned), [test])

    keep = ["game_id", "season", "game_date", "home_team", "away_team", "home_win"]
    test_predictions = tuned_p[[c for c in keep if c in tuned_p]].copy()
    test_predictions["p_coin"] = 0.5
    test_predictions["p_home_rate"] = train_rate
    test_predictions["p_elo_default"] = default_p["p_home"].to_numpy()
    test_predictions["p_elo"] = tuned_p["p_home"].to_numpy()

    y = test_predictions["home_win"].to_numpy()
    scores = pd.DataFrame(
        {
            "coin 50%": score_all(y, test_predictions["p_coin"].to_numpy()),
            "home rate (train)": score_all(y, test_predictions["p_home_rate"].to_numpy()),
            "elo default": score_all(y, test_predictions["p_elo_default"].to_numpy()),
            "elo tuned": score_all(y, test_predictions["p_elo"].to_numpy()),
        }
    ).T
    return EloExperiment(
        grid=grid,
        tuned=tuned,
        scores=scores,
        test_predictions=test_predictions.reset_index(drop=True),
        calibration=calibration_table(y, test_predictions["p_elo"].to_numpy(), n_bins=10),
    )


def home_rate_by_season(games: pd.DataFrame) -> pd.Series:
    """Home win rate per season, excluding neutral-site games."""
    real = games[~games["neutral_site"]] if "neutral_site" in games else games
    return real.groupby("season")["home_win"].mean()
