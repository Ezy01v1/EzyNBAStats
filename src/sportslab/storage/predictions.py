"""Record the models' predictions before games are played, then score them once they are.

This is the honest test of a model: predictions frozen in a database *before* the result exists
cannot leak the future. Storage is a Postgres database (Neon) whose URL comes from the
``SPORTSLAB_DATABASE_URL`` environment variable or the local, git-ignored ``.env`` file.

Row builders and the evaluation are pure pandas functions (tested without a database); the
functions that take a ``conn`` only move those frames in and out of Postgres.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.analysis.daily import add_probabilities
from sportslab.models.elo import EloParams
from sportslab.models.fantasy_model import coverage
from sportslab.models.metrics import brier_score, calibration_table, log_loss, wilson_interval

ENV_VAR = "SPORTSLAB_DATABASE_URL"
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
FANTASY_MODEL = "ridge-v1"

SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS game_predictions (
        recorded_on   date             NOT NULL,
        game_id       text             NOT NULL,
        season        text             NOT NULL,
        game_date     date             NOT NULL,
        home_team_id  integer          NOT NULL,
        away_team_id  integer          NOT NULL,
        neutral_site  boolean          NOT NULL,
        p_home        double precision NOT NULL CHECK (p_home > 0 AND p_home < 1),
        model         text             NOT NULL,
        recorded_at   timestamptz      NOT NULL DEFAULT now(),
        home_pts      integer,
        away_pts      integer,
        scored_at     timestamptz,
        PRIMARY KEY (recorded_on, game_id),
        CHECK (game_date >= recorded_on)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fantasy_predictions (
        recorded_on   date             NOT NULL,
        player_id     integer          NOT NULL,
        game_date     date             NOT NULL,
        season        text             NOT NULL,
        player        text             NOT NULL,
        team          text             NOT NULL,
        opponent      text             NOT NULL,
        home          boolean          NOT NULL,
        projection    double precision NOT NULL,
        low50         double precision NOT NULL,
        high50        double precision NOT NULL,
        low80         double precision NOT NULL,
        high80        double precision NOT NULL,
        model         text             NOT NULL,
        recorded_at   timestamptz      NOT NULL DEFAULT now(),
        actual        double precision,
        minutes       double precision,
        scored_at     timestamptz,
        PRIMARY KEY (recorded_on, player_id, game_date),
        CHECK (game_date >= recorded_on),
        CHECK (low80 <= low50 AND low50 <= high50 AND high50 <= high80)
    )
    """,
]

GAME_COLUMNS = [
    "recorded_on",
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
    "neutral_site",
    "p_home",
    "model",
]
FANTASY_COLUMNS = [
    "recorded_on",
    "player_id",
    "game_date",
    "season",
    "player",
    "team",
    "opponent",
    "home",
    "projection",
    "low50",
    "high50",
    "low80",
    "high80",
    "model",
]


# --- connection -----------------------------------------------------------------------------


def database_url(env_file: Path = ENV_FILE) -> str:
    """The database URL from the environment, else from the local ``.env`` file."""
    url = os.environ.get(ENV_VAR)
    if not url and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == ENV_VAR:
                url = value.strip().strip('"').strip("'")
    if not url:
        raise RuntimeError(f"Set {ENV_VAR} (or add it to {env_file}) to use the predictions DB.")
    return url


def connect(url: str | None = None):
    import psycopg

    return psycopg.connect(url or database_url())


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        for statement in SCHEMA_SQL:
            cur.execute(statement)
    conn.commit()


# --- row builders (pure) --------------------------------------------------------------------


def elo_label(params: EloParams) -> str:
    return f"elo k={params.k:g} ha={params.home_advantage:g} c={params.carryover:g}"


def game_prediction_rows(
    schedule: pd.DataFrame,
    ratings: dict[int, float],
    params: EloParams,
    season: str,
    recorded_on: date,
) -> pd.DataFrame:
    """Unplayed games on or after ``recorded_on`` with the current Elo probability."""
    pending = schedule[
        ~schedule["played"] & (schedule["game_date"].dt.date >= recorded_on)
    ].reset_index(drop=True)
    if pending.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)
    out = add_probabilities(pending, ratings, params.home_advantage)
    out["recorded_on"] = recorded_on
    out["season"] = season
    out["game_date"] = out["game_date"].dt.date
    out["model"] = elo_label(params)
    return out[GAME_COLUMNS]


def fantasy_prediction_rows(
    projections: pd.DataFrame, season: str, recorded_on: date, model: str = FANTASY_MODEL
) -> pd.DataFrame:
    """Projection rows for games on or after ``recorded_on`` (``project_next_games`` output)."""
    out = projections.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out = out[out["game_date"] >= recorded_on]
    out["recorded_on"] = recorded_on
    out["season"] = season
    out["model"] = model
    return out[FANTASY_COLUMNS].reset_index(drop=True)


def _records(frame: pd.DataFrame) -> list[tuple]:
    """Plain Python values (no numpy scalars) for the database driver."""
    clean = frame.astype(object).where(frame.notna(), None)
    return [
        tuple(v.item() if isinstance(v, np.generic) else v for v in row)
        for row in clean.itertuples(index=False)
    ]


# --- writes ---------------------------------------------------------------------------------


def _insert(conn, table: str, frame: pd.DataFrame, key: str) -> int:
    """Insert rows, ignoring ones already recorded today (the first snapshot of a day wins)."""
    if frame.empty:
        return 0
    cols = ", ".join(frame.columns)
    marks = ", ".join(["%s"] * len(frame.columns))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({marks}) ON CONFLICT ({key}) DO NOTHING"
    with conn.cursor() as cur:
        before = _count(cur, table)
        cur.executemany(sql, _records(frame))
        inserted = _count(cur, table) - before
    conn.commit()
    return inserted


def _count(cur, table: str) -> int:
    cur.execute(f"SELECT count(*) FROM {table}")
    return cur.fetchone()[0]


def record_game_predictions(conn, rows: pd.DataFrame) -> int:
    return _insert(conn, "game_predictions", rows, "recorded_on, game_id")


def record_fantasy_predictions(conn, rows: pd.DataFrame) -> int:
    return _insert(conn, "fantasy_predictions", rows, "recorded_on, player_id, game_date")


def score_games(conn, schedule: pd.DataFrame) -> int:
    """Fill the final score of every recorded game that has been played since."""
    pending = _pending_dates(conn, "game_predictions", "home_pts")
    played = schedule[schedule["played"] & schedule["game_date"].dt.date.isin(pending)]
    if played.empty:
        return _scored(conn, "game_predictions", "home_pts")
    params = _records(played[["home_pts", "away_pts", "game_id"]])
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE game_predictions SET home_pts = %s, away_pts = %s, scored_at = now() "
            "WHERE game_id = %s AND home_pts IS NULL",
            params,
        )
        cur.execute("SELECT count(*) FROM game_predictions WHERE home_pts IS NOT NULL")
        scored = cur.fetchone()[0]
    conn.commit()
    return scored


def score_fantasy(conn, player_games: pd.DataFrame) -> int:
    """Fill the actual fantasy points of every recorded player game that has been played."""
    games = player_games.assign(game_date=pd.to_datetime(player_games["game_date"]).dt.date)
    games = games[games["game_date"].isin(_pending_dates(conn, "fantasy_predictions", "actual"))]
    if games.empty:
        return _scored(conn, "fantasy_predictions", "actual")
    params = _records(games[["fantasy", "minutes", "player_id", "game_date"]])
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE fantasy_predictions SET actual = %s, minutes = %s, scored_at = now() "
            "WHERE player_id = %s AND game_date = %s AND actual IS NULL",
            params,
        )
        cur.execute("SELECT count(*) FROM fantasy_predictions WHERE actual IS NOT NULL")
        scored = cur.fetchone()[0]
    conn.commit()
    return scored


def _pending_dates(conn, table: str, result_column: str) -> set[date]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT game_date FROM {table} WHERE {result_column} IS NULL")
        return {row[0] for row in cur.fetchall()}


def _scored(conn, table: str, result_column: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table} WHERE {result_column} IS NOT NULL")
        return cur.fetchone()[0]


# --- reads ----------------------------------------------------------------------------------


def _read(conn, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def read_game_predictions(conn) -> pd.DataFrame:
    return _read(conn, "SELECT * FROM game_predictions ORDER BY game_date, game_id, recorded_on")


def read_fantasy_predictions(conn) -> pd.DataFrame:
    return _read(
        conn, "SELECT * FROM fantasy_predictions ORDER BY game_date, player_id, recorded_on"
    )


# --- evaluation (pure) ----------------------------------------------------------------------


def latest_before_game(predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per game: the most recent prediction recorded before (or on) its date."""
    ordered = predictions.sort_values("recorded_on")
    return ordered.drop_duplicates(keys, keep="last").reset_index(drop=True)


def evaluate_games(predictions: pd.DataFrame) -> dict:
    """Brier, log loss and calibration of the latest pre-game probability of each scored game."""
    scored = predictions.dropna(subset=["home_pts", "away_pts"])
    scored = latest_before_game(scored, ["game_id"])
    n = len(scored)
    if n == 0:
        return {"n": 0}
    y = (scored["home_pts"].astype(float) > scored["away_pts"].astype(float)).to_numpy(float)
    p = scored["p_home"].to_numpy(float)
    favorite_won = ((p >= 0.5) == (y == 1)).sum()
    low, high = wilson_interval(favorite_won, n)
    return {
        "n": n,
        "brier": brier_score(y, p),
        "brier_coin": 0.25,
        "log_loss": log_loss(y, p),
        "favorite_won": favorite_won / n,
        "favorite_won_95": (float(low), float(high)),
        "home_win_rate": float(y.mean()),
        "calibration": calibration_table(y, p, n_bins=5),
    }


def evaluate_fantasy(predictions: pd.DataFrame) -> dict:
    """Error and real interval coverage of the latest projection of each played player game."""
    scored = predictions.dropna(subset=["actual"])
    scored = latest_before_game(scored, ["player_id", "game_date"])
    n = len(scored)
    if n == 0:
        return {"n": 0}
    actual = scored["actual"].astype(float)
    out = {
        "n": n,
        "players": int(scored["player_id"].nunique()),
        "mae": float((actual - scored["projection"]).abs().mean()),
        "bias": float((actual - scored["projection"]).mean()),
    }
    for level in (50, 80):
        cov = coverage(actual, scored[f"low{level}"], scored[f"high{level}"])
        low, high = wilson_interval(cov * n, n)
        out[f"coverage{level}"] = cov
        out[f"coverage{level}_95"] = (float(low), float(high))
    return out
