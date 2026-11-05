import uuid
from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportslab.models.elo import EloParams
from sportslab.storage import predictions as store

PARAMS = EloParams(k=24, home_advantage=40, carryover=0.7)
NOON_NOV4 = pd.Timestamp("2026-11-04 17:00", tz="UTC")  # 11:00 in Mexico City


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4", "g5"],
            "game_date": pd.to_datetime(
                ["2026-11-03", "2026-11-04", "2026-11-04", "2026-11-05", "2026-11-04"]
            ),
            "played": [True, True, False, False, False],
            "home_team_id": [1, 3, 1, 2, 5],
            "away_team_id": [2, 4, 3, 4, 6],
            "home_pts": [110, 99, 0, 0, 0],
            "away_pts": [100, 101, 0, 0, 0],
            "neutral_site": [False, False, False, True, False],
            "tipoff_utc": pd.to_datetime(
                [
                    "2026-11-04 00:00",
                    "2026-11-04 16:00",
                    "2026-11-05 00:00",
                    "2026-11-06 00:00",
                    "2026-11-04 16:30",  # g5: in progress at NOON_NOV4
                ],
                utc=True,
            ),
        }
    )


def _projections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g3", "g0", "g5"],
            "player_id": np.array([7, 8, 9], dtype=np.int64),
            "player": ["A", "B", "C"],
            "team": ["BOS", "NYK", "LAL"],
            "opponent": ["NYK", "BOS", "GSW"],
            "home": [True, False, True],
            "game_date": pd.to_datetime(["2026-11-04", "2026-11-03", "2026-11-04"]),
            "n_prior_season": [5, 5, 5],
            "season_mean": [30.0, 20.0, 25.0],
            "projection": [31.0, 22.0, 26.0],
            "low50": [25.0, 17.0, 20.0],
            "high50": [37.0, 27.0, 32.0],
            "low80": [19.0, 11.0, 14.0],
            "high80": [44.0, 33.0, 38.0],
        }
    )


def _games(day=date(2026, 11, 4), now=NOON_NOV4, schedule=None):
    return store.game_prediction_rows(
        _schedule() if schedule is None else schedule,
        {1: 1600.0, 3: 1500.0},
        PARAMS,
        "2026-27",
        day,
        now=now,
    )


# --- row builders ---------------------------------------------------------------------------


def test_only_pregame_unplayed_games_from_today_on_are_recorded():
    rows = _games()
    assert list(rows["game_id"]) == ["g3", "g4"]  # g1/g2 played, g5 already tipped off
    assert list(rows.columns) == store.GAME_COLUMNS
    assert rows.loc[0, "p_home"] > 0.5  # stronger side at home
    assert rows.loc[1, "p_home"] == pytest.approx(0.5)  # neutral site, equal default ratings
    assert rows.loc[0, "game_date"] == date(2026, 11, 4)
    assert rows.loc[0, "model"] == "elo k=24 ha=40 c=0.7"
    assert rows.loc[0, "tipoff_at"] == pd.Timestamp("2026-11-05 00:00", tz="UTC")


def test_a_late_catch_up_run_skips_every_game_already_started():
    late = pd.Timestamp("2026-11-05 01:00", tz="UTC")  # the PC was off until 19:00 local
    assert list(_games(now=late)["game_id"]) == ["g4"]


def test_games_without_a_known_tipoff_are_kept():
    schedule = _schedule().assign(tipoff_utc=pd.NaT)
    assert list(_games(schedule=schedule)["game_id"]) == ["g3", "g4", "g5"]
    assert _games(schedule=schedule.drop(columns="tipoff_utc"))["tipoff_at"].isna().all()


def test_no_pending_games_gives_an_empty_frame_with_the_columns():
    rows = _games(day=date(2027, 1, 1))
    assert rows.empty and list(rows.columns) == store.GAME_COLUMNS


def test_fantasy_rows_drop_past_games_and_games_in_progress():
    rows = store.fantasy_prediction_rows(
        _projections(), "2026-27", date(2026, 11, 4), started_games={"g5"}
    )
    assert list(rows["player_id"]) == [7]
    assert list(rows.columns) == store.FANTASY_COLUMNS
    assert rows.loc[0, "model"] == store.FANTASY_MODEL and rows.loc[0, "game_id"] == "g3"


def test_records_are_plain_python_values_and_missing_values_are_none():
    rows = _games(schedule=_schedule().assign(tipoff_utc=pd.NaT))
    record = store._records(rows)[0]
    assert all(not isinstance(v, np.generic) for v in record)
    assert record[-1] is None  # unknown tip-off goes to the database as NULL


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


# --- evaluation -----------------------------------------------------------------------------


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


# --- against the real database, in a throwaway schema ----------------------------------------


@pytest.fixture
def db():
    """A connection whose tables live in a new schema, dropped afterwards (real tables safe)."""
    psycopg = pytest.importorskip("psycopg")
    try:
        direct = store.database_url().replace("-pooler", "")  # the pooler drops session settings
    except RuntimeError:
        pytest.skip("no database configured")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(direct, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    try:
        with psycopg.connect(direct, options=f"-c search_path={schema}") as conn:
            store.ensure_schema(conn)
            yield conn
            conn.rollback()
    finally:
        with psycopg.connect(direct, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")


def _logs(rows):
    return pd.DataFrame(rows, columns=["game_id", "player_id", "game_date", "fantasy", "minutes"])


@pytest.mark.network
def test_db_record_twice_then_score_and_read_back(db):
    games = _games()
    assert store.record_game_predictions(db, games) == 2
    assert store.record_game_predictions(db, games) == 0  # same day: no duplicates
    fantasy = store.fantasy_prediction_rows(_projections(), "2026-27", date(2026, 11, 4))
    assert store.record_fantasy_predictions(db, fantasy) == 2  # g3 and g5 (no filter given)

    finished = _schedule().assign(played=True, home_pts=[110, 99, 120, 90, 100])
    assert store.score_games(db, finished) == 2
    assert store.score_games(db, finished) == 2  # scoring again changes nothing
    logs = _logs([("g3", 7, pd.Timestamp("2026-11-04"), 33.5, 34.0)])
    assert store.score_fantasy(db, logs) == 1  # player 9 did not play: stays unscored
    read = store.read_game_predictions(db)
    assert set(read["game_id"]) == {"g3", "g4"} and read["home_pts"].notna().all()
    assert read["tipoff_at"].notna().all()
    stored = store.read_fantasy_predictions(db).set_index("player_id")
    assert stored.loc[7, "actual"] == 33.5 and pd.isna(stored.loc[9, "actual"])


@pytest.mark.network
def test_db_a_postponed_game_is_still_scored_by_its_id(db):
    store.record_game_predictions(db, _games())  # g3 recorded for 2026-11-04
    store.record_fantasy_predictions(
        db, store.fantasy_prediction_rows(_projections(), "2026-27", date(2026, 11, 4))
    )
    moved = _schedule()
    moved.loc[moved["game_id"] == "g3", ["game_date", "played", "home_pts", "away_pts"]] = [
        pd.Timestamp("2026-12-01"),
        True,
        101,
        97,
    ]
    assert store.score_games(db, moved) == 1
    logs = _logs([("g3", 7, pd.Timestamp("2026-12-01"), 28.0, 30.0)])  # played on the new date
    assert store.score_fantasy(db, logs) == 1
    g3 = store.read_game_predictions(db).set_index("game_id").loc["g3"]
    assert (g3["home_pts"], g3["away_pts"]) == (101, 97)


@pytest.mark.network
def test_db_rows_recorded_before_game_ids_existed_fall_back_to_the_date(db):
    rows = store.fantasy_prediction_rows(_projections(), "2026-27", date(2026, 11, 4))
    store.record_fantasy_predictions(db, rows.assign(game_id=None))
    logs = _logs([("g3", 7, pd.Timestamp("2026-11-04"), 30.0, 31.0)])
    assert store.score_fantasy(db, logs) == 1


@pytest.mark.network
def test_db_several_missed_days_are_scored_in_one_catch_up_run(db):
    for day, now in (
        (date(2026, 11, 4), NOON_NOV4),
        (date(2026, 11, 5), pd.Timestamp("2026-11-05 17:00", tz="UTC")),
    ):
        store.record_game_predictions(db, _games(day=day, now=now))
    finished = _schedule().assign(played=True, home_pts=[110, 99, 120, 90, 100])
    assert store.score_games(db, finished) == 3  # g3 (Nov 4), g4 twice (Nov 4 and Nov 5)
    result = store.evaluate_games(store.read_game_predictions(db))
    assert result["n"] == 2  # one prediction per game: the latest pre-game one


@pytest.mark.network
def test_db_two_models_share_a_game_and_old_tables_are_migrated(db):
    with db.cursor() as cur:  # recreate the table with the original key (recorded_on, game_id)
        cur.execute("ALTER TABLE game_predictions DROP CONSTRAINT game_predictions_pkey")
        cur.execute("ALTER TABLE game_predictions ADD PRIMARY KEY (recorded_on, game_id)")
    db.commit()
    store.ensure_schema(db)  # the migration widens the key to include the model
    store.ensure_schema(db)  # and running it again is harmless
    elo = _games()
    other = elo.assign(model="logistic-form-rest-v1", p_home=0.55)
    assert store.record_game_predictions(db, elo) == 2
    assert store.record_game_predictions(db, other) == 2
    assert store.record_game_predictions(db, other) == 0
    finished = _schedule().assign(played=True, home_pts=[110, 99, 120, 90, 100])
    assert store.score_games(db, finished) == 4
    by_model = store.evaluate_games_by_model(store.read_game_predictions(db))
    assert set(by_model) == {"elo k=24 ha=40 c=0.7", "logistic-form-rest-v1"}
    assert all(r["n"] == 2 for r in by_model.values())
