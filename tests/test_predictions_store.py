import uuid
from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportslab.models.elo import EloParams
from sportslab.storage import predictions as store

PARAMS = EloParams(k=24, home_advantage=40, carryover=0.7)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "game_date": pd.to_datetime(["2026-11-03", "2026-11-04", "2026-11-04", "2026-11-05"]),
            "played": [True, True, False, False],
            "home_team_id": [1, 3, 1, 2],
            "away_team_id": [2, 4, 3, 4],
            "home_pts": [110, 99, 0, 0],
            "away_pts": [100, 101, 0, 0],
            "neutral_site": [False, False, False, True],
        }
    )


def _projections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": np.array([7, 8], dtype=np.int64),
            "player": ["A", "B"],
            "team": ["BOS", "NYK"],
            "opponent": ["NYK", "BOS"],
            "home": [True, False],
            "game_date": pd.to_datetime(["2026-11-04", "2026-11-03"]),
            "n_prior_season": [5, 5],
            "season_mean": [30.0, 20.0],
            "projection": [31.0, 22.0],
            "low50": [25.0, 17.0],
            "high50": [37.0, 27.0],
            "low80": [19.0, 11.0],
            "high80": [44.0, 33.0],
        }
    )


def test_only_unplayed_games_from_today_on_are_recorded():
    rows = store.game_prediction_rows(
        _schedule(), {1: 1600.0, 3: 1500.0}, PARAMS, "2026-27", date(2026, 11, 4)
    )
    assert list(rows["game_id"]) == ["g3", "g4"]
    assert list(rows.columns) == store.GAME_COLUMNS
    assert rows.loc[0, "p_home"] > 0.5  # stronger side at home
    assert rows.loc[1, "p_home"] == pytest.approx(0.5)  # neutral site, equal default ratings
    assert rows.loc[0, "game_date"] == date(2026, 11, 4)
    assert rows.loc[0, "model"] == "elo k=24 ha=40 c=0.7"


def test_no_pending_games_gives_an_empty_frame_with_the_columns():
    rows = store.game_prediction_rows(_schedule(), {}, PARAMS, "2026-27", date(2027, 1, 1))
    assert rows.empty and list(rows.columns) == store.GAME_COLUMNS


def test_fantasy_rows_drop_games_before_the_recording_day():
    rows = store.fantasy_prediction_rows(_projections(), "2026-27", date(2026, 11, 4))
    assert list(rows["player_id"]) == [7]
    assert list(rows.columns) == store.FANTASY_COLUMNS
    assert rows.loc[0, "model"] == store.FANTASY_MODEL


def test_records_are_plain_python_values():
    rows = store.fantasy_prediction_rows(_projections(), "2026-27", date(2026, 11, 3))
    record = store._records(rows)[0]
    assert all(not isinstance(v, np.generic) for v in record)


def test_database_url_prefers_the_environment_then_the_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"OTHER=1\n{store.ENV_VAR}='postgresql://file'\n", encoding="utf-8")
    monkeypatch.delenv(store.ENV_VAR, raising=False)
    assert store.database_url(env) == "postgresql://file"
    monkeypatch.setenv(store.ENV_VAR, "postgresql://env")
    assert store.database_url(env) == "postgresql://env"
    monkeypatch.delenv(store.ENV_VAR)
    with pytest.raises(RuntimeError):
        store.database_url(tmp_path / "missing.env")


def test_game_evaluation_uses_the_latest_pregame_prediction():
    preds = pd.DataFrame(
        {
            "recorded_on": [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 2)],
            "game_id": ["g1", "g1", "g2"],
            "p_home": [0.2, 0.8, 0.6],
            "home_pts": [110, 110, 90],
            "away_pts": [100, 100, 95],
        }
    )
    result = store.evaluate_games(preds)
    assert result["n"] == 2
    # g1 uses 0.8 (home won), g2 uses 0.6 (home lost)
    assert result["brier"] == pytest.approx(((0.8 - 1) ** 2 + 0.6**2) / 2)
    assert result["favorite_won"] == pytest.approx(0.5)


def test_unscored_predictions_are_not_evaluated():
    preds = pd.DataFrame(
        {
            "recorded_on": [date(2026, 11, 1)],
            "game_id": ["g1"],
            "p_home": [0.6],
            "home_pts": [None],
            "away_pts": [None],
        }
    )
    assert store.evaluate_games(preds) == {"n": 0}
    empty = pd.DataFrame(columns=["recorded_on", "player_id", "game_date", "actual"])
    assert store.evaluate_fantasy(empty) == {"n": 0}


def test_fantasy_evaluation_reports_error_and_real_coverage():
    preds = pd.DataFrame(
        {
            "recorded_on": [date(2026, 11, 1)] * 4,
            "player_id": [1, 2, 3, 4],
            "game_date": [date(2026, 11, 2)] * 4,
            "projection": [20.0] * 4,
            "low50": [15.0] * 4,
            "high50": [25.0] * 4,
            "low80": [10.0] * 4,
            "high80": [30.0] * 4,
            "actual": [20.0, 27.0, 35.0, None],  # player 4 did not play
        }
    )
    result = store.evaluate_fantasy(preds)
    assert result["n"] == 3
    assert result["coverage50"] == pytest.approx(1 / 3)
    assert result["coverage80"] == pytest.approx(2 / 3)
    assert result["mae"] == pytest.approx((0 + 7 + 15) / 3)
    lo, hi = result["coverage80_95"]
    assert lo < 2 / 3 < hi


@pytest.mark.network
def test_round_trip_against_the_real_database_in_a_throwaway_schema():
    """Record, re-record (no duplicates), score and read back, then drop the schema."""
    import psycopg

    schema = f"test_{uuid.uuid4().hex[:8]}"
    direct = store.database_url().replace("-pooler", "")  # the pooler drops session settings
    with psycopg.connect(direct, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    with psycopg.connect(direct, options=f"-c search_path={schema}") as conn:
        try:
            store.ensure_schema(conn)
            today = date(2026, 11, 4)
            games = store.game_prediction_rows(_schedule(), {}, PARAMS, "2026-27", today)
            assert store.record_game_predictions(conn, games) == 2
            assert store.record_game_predictions(conn, games) == 0  # same day: no duplicates
            fantasy = store.fantasy_prediction_rows(_projections(), "2026-27", today)
            assert store.record_fantasy_predictions(conn, fantasy) == 1

            finished = _schedule().assign(played=True, home_pts=[110, 99, 120, 90])
            assert store.score_games(conn, finished) == 2
            logs = pd.DataFrame(
                {
                    "player_id": [7],
                    "game_date": pd.to_datetime(["2026-11-04"]),
                    "fantasy": [33.5],
                    "minutes": [34.0],
                }
            )
            assert store.score_fantasy(conn, logs) == 1
            read = store.read_game_predictions(conn)
            assert set(read["game_id"]) == {"g3", "g4"} and read["home_pts"].notna().all()
            assert store.read_fantasy_predictions(conn)["actual"].tolist() == [33.5]
        finally:
            conn.rollback()
    with psycopg.connect(direct, autocommit=True) as admin:
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
