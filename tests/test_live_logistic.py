from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportslab.features.team_form import build_pregame_features
from sportslab.models.elo import EloParams
from sportslab.models.live_logistic import (
    LOGISTIC_MODEL,
    fit_logistic,
    pending_features,
    predict_pending,
)
from sportslab.storage import predictions as store

PARAMS = EloParams(k=24, home_advantage=40, carryover=0.7)


def _schedule() -> pd.DataFrame:
    # Team 1 plays every day; 2-5 alternate. Days 1-4 played, days 5-6 pending.
    days = pd.to_datetime([f"2026-11-0{d}" for d in (1, 2, 3, 4, 5, 6)])
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(6)],
            "game_date": days,
            "played": [True] * 4 + [False] * 2,
            "home_team_id": [1, 3, 1, 5, 1, 2],
            "away_team_id": [2, 1, 4, 1, 3, 4],
            "home_pts": [110, 90, 120, 100, 0, 0],
            "away_pts": [100, 100, 100, 104, 0, 0],
            "neutral_site": [False, False, False, False, False, True],
        }
    )


def test_pending_rows_get_rest_form_and_elo_from_known_information():
    pending = pending_features(_schedule(), "2026-27", {1: 1600.0, 3: 1500.0})
    assert list(pending["game_id"]) == ["g4", "g5"]
    g4 = pending.iloc[0]
    assert g4["home_rest"] == 1 and g4["home_b2b"] == 1  # team 1 played the day before
    assert g4["away_rest"] == 3  # team 3 last played on Nov 2
    # team 1 margins so far: +10, +10, +20, +4 -> 44 / (4 + 5)
    assert g4["home_form"] == pytest.approx(44 / 9)
    assert g4["elo_diff"] == pytest.approx(100.0)


def test_pending_features_ignore_placeholder_scores():
    a = pending_features(_schedule(), "2026-27", {})
    s = _schedule()
    s.loc[4, ["home_pts", "away_pts"]] = [150, 50]  # garbage in an unplayed row
    b = pending_features(s, "2026-27", {})
    pd.testing.assert_frame_equal(
        a.drop(columns=["home_pts", "away_pts"]), b.drop(columns=["home_pts", "away_pts"])
    )


def _history(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    home = rng.integers(1, 11, n)
    away = (home + rng.integers(1, 10, n) - 1) % 10 + 1
    strength = np.arange(11) * 3.0
    margin = strength[home] - strength[away] + 3 + rng.normal(0, 12, n)
    pts_home = 100 + np.round(margin / 2)
    pts_away = 100 - np.round(margin / 2)
    pts_away = np.where(pts_home == pts_away, pts_away - 1, pts_away)
    return pd.DataFrame(
        {
            "game_id": [f"h{i}" for i in range(n)],
            "season": "2025-26",
            "game_date": pd.Timestamp("2025-10-20") + pd.to_timedelta(np.arange(n) // 5, "D"),
            "home_team_id": home,
            "away_team_id": away,
            "home_pts": pts_home,
            "away_pts": pts_away,
            "home_win": (pts_home > pts_away).astype(int),
            "neutral_site": False,
        }
    )


def test_neutral_games_are_skipped_and_rows_fit_the_log():
    model = fit_logistic(build_pregame_features(_history(), PARAMS))
    predicted = predict_pending(model, pending_features(_schedule(), "2026-27", {1: 1600.0}))
    assert list(predicted["game_id"]) == ["g4"]  # g5 is neutral
    assert 0.5 < predicted.loc[0, "p_home"] < 1  # stronger home side
    now = pd.Timestamp("2026-11-05 12:00", tz="UTC")
    rows = store.model_game_rows(predicted, LOGISTIC_MODEL, "2026-27", date(2026, 11, 5), now)
    assert list(rows.columns) == store.GAME_COLUMNS
    assert rows.loc[0, "model"] == LOGISTIC_MODEL


def test_compare_models_pairs_the_same_games():
    preds = pd.DataFrame(
        {
            "recorded_on": [date(2026, 11, 1)] * 4,
            "game_id": ["g1", "g2", "g1", "g2"],
            "model": ["a", "a", "b", "b"],
            "p_home": [0.7, 0.4, 0.5, 0.5],
            "home_pts": [110, 90, 110, 90],
            "away_pts": [100, 95, 100, 95],
        }
    )
    result = store.compare_models(preds, "a", "b")
    assert result["n"] == 2 and result["gain"] > 0  # "a" leaned the right way both times
    by_model = store.evaluate_games_by_model(preds)
    assert by_model["b"]["brier"] == pytest.approx(0.25)
    assert store.compare_models(preds, "a", "missing") == {"n": 0}
