import numpy as np
import pandas as pd
import pytest

from sportslab.features.fantasy import FEATURES, fill_missing
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


def _rows(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true = rng.uniform(10, 40, n)  # each row is a different player with a stable true level
    games = rng.integers(0, 40, n)
    noise = rng.normal(0, 8, n)
    prior = true + rng.normal(0, 3, n)  # last season's mean is a good guide to this season's level
    season_mean = np.where(
        games > 0, true + rng.normal(0, 8, n) / np.sqrt(np.maximum(games, 1)), np.nan
    )
    df = pd.DataFrame(
        {
            "target": np.clip(true + noise, 0, None),
            "season_mean": season_mean,
            "n_prior_season": games,
            "prior_season_mean": prior,
            "last5": true + rng.normal(0, 8, n) / np.sqrt(5),
            "last10_ewm": true + rng.normal(0, 8, n) / np.sqrt(8),
            "min_last5": rng.uniform(20, 36, n),
            "home": rng.integers(0, 2, n),
            "rest_days": rng.integers(1, 5, n),
            "b2b": rng.integers(0, 2, n),
            "opp_allowed_rel": rng.normal(0, 1, n),
            "player_id": np.arange(n) % 50,
        }
    )
    return df


def test_baseline_switches_from_last_season_to_this_season_after_ten_games():
    rows = _rows()
    base = baseline_season(rows)
    early, late = rows["n_prior_season"] < 10, rows["n_prior_season"] >= 10
    assert (base[early] == rows.loc[early, "prior_season_mean"]).all()
    assert (base[late] == rows.loc[late, "season_mean"]).all()
    assert baseline_last5(rows).equals(rows["last5"])


def test_shrunk_mean_is_last_season_with_no_games_and_this_season_with_many():
    rows = pd.DataFrame(
        {
            "season_mean": [np.nan, 30.0],
            "n_prior_season": [0, 1000],
            "prior_season_mean": [20.0, 10.0],
        }
    )
    out = shrunk_mean(rows, k=10)
    assert out.iloc[0] == pytest.approx(20.0) and out.iloc[1] == pytest.approx(30.0, abs=0.2)


def test_the_best_shrinkage_is_large_when_this_seasons_start_is_noisy():
    rows = _rows(n=3000, seed=1)
    k = fit_shrinkage(rows, [0, 5, 20, 80, 320])
    assert k > 0  # noisy early-season means: leaning on last season helps


def test_ridge_beats_the_noisy_last5_on_fresh_data():
    train, test = fill_missing(_rows(n=2000, seed=2)), fill_missing(_rows(n=800, seed=3))
    model = fit_ridge(train)
    ridge_mae = np.abs(test["target"] - predict_ridge(model, test)).mean()
    last5_mae = np.abs(test["target"] - baseline_last5(test)).mean()
    assert ridge_mae < last5_mae and set(FEATURES) <= set(test.columns)


def test_intervals_have_the_right_coverage_on_new_data_and_grow_with_the_level():
    rng = np.random.default_rng(4)

    def make(n):
        level = rng.uniform(10, 40, n)
        return level, level + rng.normal(0, 1, n) * (0.15 * level)  # spread grows with the level

    p_cal, y_cal = make(6000)
    p_new, y_new = make(4000)
    cal = IntervalCalibrator().fit(p_cal, y_cal)
    for level in (0.5, 0.8, 0.9):
        low, high = cal.interval(p_new, level)
        assert coverage(y_new, low, high) == pytest.approx(level, abs=0.04)
    low, high = cal.interval(np.array([12.0, 38.0]), 0.8)
    assert (high - low)[1] > 2 * (high - low)[0] and (low >= 0).all()


def test_paired_gain_is_positive_for_a_better_predictor_and_resamples_players():
    rng = np.random.default_rng(5)
    n = 600
    players = pd.Series(np.arange(n) % 40)
    actual = pd.Series(rng.normal(25, 6, n))
    better = actual + rng.normal(0, 1, n)
    worse = actual + rng.normal(0, 6, n)
    gain, low, high = paired_abs_error_gain(actual, better, worse, players, n_boot=200)
    assert gain > 0 and low > 0 and low <= gain <= high
    zero = paired_abs_error_gain(actual, worse, worse, players, n_boot=50)
    assert zero == pytest.approx((0, 0, 0))
