import numpy as np
import pandas as pd
import pytest

from sportslab.features.team_form import build_pregame_features
from sportslab.models.elo import EloParams
from sportslab.models.metrics import paired_log_loss_gain
from sportslab.models.supervised import ALL_FEATURES, run_supervised_experiment

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def _synthetic_features(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 4, size=8)
    rows = []
    for s, season in enumerate(SEASONS):
        start = pd.Timestamp("2019-10-20") + pd.Timedelta(days=365 * s)
        for day in range(90):
            teams = rng.choice(8, size=4, replace=False)
            for home, away in (teams[:2], teams[2:]):
                margin = strength[home] + 2 - strength[away] + rng.normal(0, 12)
                rows.append(
                    {
                        "game_date": start + pd.Timedelta(days=day),
                        "season": season,
                        "home_team_id": int(home),
                        "away_team_id": int(away),
                        "home_pts": 110 + margin / 2,
                        "away_pts": 110 - margin / 2,
                    }
                )
    games = pd.DataFrame(rows)
    games["game_id"] = [f"{i:05d}" for i in range(len(games))]
    games["home_team"] = games["home_team_id"].astype(str)
    games["away_team"] = games["away_team_id"].astype(str)
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
    games["neutral_site"] = False
    return build_pregame_features(games, EloParams())


KWARGS = {
    "train": ["2019-20", "2020-21", "2021-22", "2022-23"],
    "validation": ["2023-24"],
    "test": "2024-25",
}


def test_experiment_reports_every_model_on_validation_and_test():
    exp = run_supervised_experiment(_synthetic_features(), **KWARGS)
    expected = {"home rate", "elo", "logistic (elo only)", "logistic (all)", "xgboost (all)"}
    assert set(exp.scores_validation.index) == expected
    assert set(exp.scores_test.index) == expected
    assert {"log_loss", "brier", "accuracy"} <= set(exp.scores_test.columns)
    assert set(exp.test_predictions["season"]) == {"2024-25"}
    assert exp.best_iteration >= 1
    assert set(exp.logistic_coefficients.index) == set(ALL_FEATURES)


def test_experiment_gains_vs_elo_carry_a_confidence_interval():
    exp = run_supervised_experiment(_synthetic_features(), **KWARGS)
    gains = exp.gain_vs_elo_test
    assert {"gain", "ci_low", "ci_high"} <= set(gains.columns)
    assert (gains["ci_low"] <= gains["gain"]).all() and (gains["gain"] <= gains["ci_high"]).all()


def test_test_labels_never_influence_model_selection_or_validation_scores():
    feats = _synthetic_features()
    flipped = feats.copy()
    test = flipped["season"] == "2024-25"
    flipped.loc[test, "home_win"] = 1 - flipped.loc[test, "home_win"]
    a = run_supervised_experiment(feats, **KWARGS)
    b = run_supervised_experiment(flipped, **KWARGS)
    assert a.best_iteration == b.best_iteration
    pd.testing.assert_frame_equal(a.scores_validation, b.scores_validation)


def test_paired_gain_is_positive_when_the_new_model_is_better():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0] * 20)
    spread = np.linspace(0.6, 0.9, len(y))  # varying confidence, so per-game gains differ
    better = np.where(y == 1, spread, 1 - spread)
    worse = np.full(len(y), 0.5)
    gain, low, high = paired_log_loss_gain(y, better, worse)
    assert gain > 0 and low > 0 and low < gain < high


def test_paired_gain_interval_straddles_zero_for_identical_models():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.6, 0.4, 0.55, 0.45])
    gain, low, high = paired_log_loss_gain(y, p, p)
    assert gain == pytest.approx(0) and low == pytest.approx(0) and high == pytest.approx(0)
