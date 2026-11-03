import numpy as np
import pytest

from sportslab.models.metrics import accuracy, brier_score, calibration_table, log_loss


def test_log_loss_of_coin_flip_is_ln2():
    y = np.array([1, 0, 1, 0])
    assert log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))


def test_log_loss_rewards_confident_correct_and_punishes_confident_wrong():
    y = np.array([1, 0])
    assert log_loss(y, np.array([0.9, 0.1])) < log_loss(y, np.array([0.6, 0.4]))
    assert log_loss(y, np.array([0.1, 0.9])) > np.log(2)


def test_log_loss_does_not_explode_on_certain_wrong_prediction():
    assert np.isfinite(log_loss(np.array([1]), np.array([0.0])))


def test_brier_score():
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert brier_score(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_accuracy_uses_threshold_half():
    assert accuracy(np.array([1, 0, 1]), np.array([0.7, 0.4, 0.2])) == pytest.approx(2 / 3)


def test_calibration_table_bins_predictions():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95, 0.95])
    table = calibration_table(y, p, n_bins=10)
    assert list(table["n"]) == [2, 3]
    assert table["mean_pred"].iloc[0] == pytest.approx(0.05)
    assert table["observed"].iloc[0] == 1.0
    assert table["observed"].iloc[1] == pytest.approx(1 / 3)


def test_accuracy_counts_a_dead_even_prediction_as_half_right():
    assert accuracy(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_calibration_table_has_wilson_interval_around_observed_rate():
    table = calibration_table(np.array([1, 0, 1, 1, 0]), np.full(5, 0.6), n_bins=10)
    row = table.iloc[0]
    assert 0 <= row["ci_low"] < row["observed"] < row["ci_high"] <= 1
