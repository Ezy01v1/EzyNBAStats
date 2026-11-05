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
from sportslab.models.metrics import (
    brier_score,
    calibration_table,
    log_loss,
    paired_log_loss_gain,
    wilson_interval,
)

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
    # Added 2026-11-04 (after the first snapshot): tip-off time and the fantasy game id, which
    # lets postponed games be scored. Older rows have NULL here.
    "ALTER TABLE game_predictions ADD COLUMN IF NOT EXISTS tipoff_at timestamptz",
    "ALTER TABLE fantasy_predictions ADD COLUMN IF NOT EXISTS game_id text",
    # Added 2026-11-04: several game models side by side, one row per (day, game, model).
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.key_column_usage
            WHERE table_schema = current_schema() AND table_name = 'game_predictions'
              AND constraint_name = 'game_predictions_pkey' AND column_name = 'model'
        ) THEN
            ALTER TABLE game_predictions DROP CONSTRAINT game_predictions_pkey;
            ALTER TABLE game_predictions ADD PRIMARY KEY (recorded_on, game_id, model);
        END IF;
    END $$
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
    "tipoff_at",
]
FANTASY_COLUMNS = [
    "recorded_on",
    "game_id",
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
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Unplayed games on or after ``recorded_on`` that have not tipped off, with their Elo
    probability. A game already in progress is left out: its prediction would not be pre-game.
    """
    pending = schedule[pregame(schedule, recorded_on, now)].reset_index(drop=True)
    if pending.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)
    out = add_probabilities(pending, ratings, params.home_advantage)
    return _game_rows(out, season, recorded_on, elo_label(params))


def model_game_rows(
    predicted: pd.DataFrame,
    model: str,
    season: str,
    recorded_on: date,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rows for any model's ``p_home`` (schedule columns + ``p_home``), same filters as Elo."""
    out = predicted[pregame(predicted, recorded_on, now)].reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)
    return _game_rows(out, season, recorded_on, model)


def pregame(schedule: pd.DataFrame, recorded_on: date, now: pd.Timestamp | None = None):
    """Unplayed games on or after ``recorded_on`` that have not tipped off."""
    keep = ~schedule["played"] & (pd.to_datetime(schedule["game_date"]).dt.date >= recorded_on)
    return keep & ~started(schedule, now)


def _game_rows(out: pd.DataFrame, season: str, recorded_on: date, model: str) -> pd.DataFrame:
    out = out.copy()
    out["recorded_on"] = recorded_on
    out["season"] = season
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out["model"] = model
    out["tipoff_at"] = _tipoffs(out)
    return out[GAME_COLUMNS]


def _tipoffs(schedule: pd.DataFrame) -> pd.Series:
    if "tipoff_utc" not in schedule:
        return pd.Series(pd.NaT, index=schedule.index, dtype="datetime64[us, UTC]")
    return pd.to_datetime(schedule["tipoff_utc"], utc=True)


def started(schedule: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.Series:
    """True for games whose tip-off time has passed (False when the time is unknown)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return (_tipoffs(schedule) <= now).fillna(False).astype(bool)


def fantasy_prediction_rows(
    projections: pd.DataFrame,
    season: str,
    recorded_on: date,
    model: str = FANTASY_MODEL,
    started_games: set[str] | None = None,
) -> pd.DataFrame:
    """Projection rows for games on or after ``recorded_on`` (``project_next_games`` output),
    leaving out the games in ``started_games`` (ids of games already in progress)."""
    out = projections.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out = out[out["game_date"] >= recorded_on]
    if started_games:
        out = out[~out["game_id"].isin(started_games)]
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
    return _insert(conn, "game_predictions", rows, "recorded_on, game_id, model")


def record_fantasy_predictions(conn, rows: pd.DataFrame) -> int:
    return _insert(conn, "fantasy_predictions", rows, "recorded_on, player_id, game_date")


def score_games(conn, schedule: pd.DataFrame) -> int:
    """Fill the final score of every recorded game that has been played since.

    Matched by game id, so a game that was postponed to another date is still scored.
    """
    pending = _pending_values(conn, "game_predictions", "game_id", "home_pts")
    played = schedule[schedule["played"] & schedule["game_id"].isin(pending)]
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
    """Fill the actual fantasy points of every recorded player game that has been played.

    Matched by (player, game id); rows recorded before the game id was stored fall back to
    (player, date). A player who did not play stays unscored (projections are "if he plays").
    """
    games = player_games.assign(game_date=pd.to_datetime(player_games["game_date"]).dt.date)
    by_id = games[
        games["game_id"].isin(_pending_values(conn, "fantasy_predictions", "game_id", "actual"))
    ]
    by_date = games[games["game_date"].isin(_pending_dates_without_id(conn))]
    with conn.cursor() as cur:
        if not by_id.empty:
            cur.executemany(
                "UPDATE fantasy_predictions SET actual = %s, minutes = %s, scored_at = now() "
                "WHERE player_id = %s AND game_id = %s AND actual IS NULL",
                _records(by_id[["fantasy", "minutes", "player_id", "game_id"]]),
            )
        if not by_date.empty:
            cur.executemany(
                "UPDATE fantasy_predictions SET actual = %s, minutes = %s, scored_at = now() "
                "WHERE player_id = %s AND game_date = %s AND game_id IS NULL AND actual IS NULL",
                _records(by_date[["fantasy", "minutes", "player_id", "game_date"]]),
            )
    conn.commit()
    return _scored(conn, "fantasy_predictions", "actual")


def _pending_values(conn, table: str, column: str, result_column: str) -> set:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT DISTINCT {column} FROM {table} "
            f"WHERE {result_column} IS NULL AND {column} IS NOT NULL"
        )
        return {row[0] for row in cur.fetchall()}


def _pending_dates_without_id(conn) -> set[date]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT game_date FROM fantasy_predictions "
            "WHERE actual IS NULL AND game_id IS NULL"
        )
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


def _scored_games(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.dropna(subset=["home_pts", "away_pts"])
    keys = ["model", "game_id"] if "model" in scored else ["game_id"]
    return latest_before_game(scored, keys)


def evaluate_games(predictions: pd.DataFrame) -> dict:
    """Brier, log loss and calibration of the latest pre-game probability of each scored game.

    Pass one model's rows (see ``evaluate_games_by_model``)."""
    scored = _scored_games(predictions)
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


def evaluate_games_by_model(predictions: pd.DataFrame) -> dict[str, dict]:
    return {m: evaluate_games(rows) for m, rows in predictions.groupby("model")}


def compare_models(predictions: pd.DataFrame, model: str, baseline: str) -> dict:
    """Paired log-loss gain of ``model`` over ``baseline`` on the scored games both predicted
    (positive = ``model`` better), with a 95% interval."""
    scored = _scored_games(predictions)
    wide = scored.pivot(index="game_id", columns="model", values="p_home")
    if model not in wide or baseline not in wide:
        return {"n": 0}
    wide = wide[[model, baseline]].dropna()
    if wide.empty:
        return {"n": 0}
    result = scored.drop_duplicates("game_id").set_index("game_id").loc[wide.index]
    y = (result["home_pts"].astype(float) > result["away_pts"].astype(float)).to_numpy(float)
    gain, low, high = paired_log_loss_gain(y, wide[model].to_numpy(), wide[baseline].to_numpy())
    return {"n": len(wide), "gain": float(gain), "gain_95": (float(low), float(high))}


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


# --- end-of-season pruning ------------------------------------------------------------------

# Rows kept per key: the latest recorded before the game, i.e. exactly what the evaluation uses.
PRUNE_KEYS = {
    "game_predictions": ["game_id", "model"],
    "fantasy_predictions": ["player_id", "game_date", "model"],
}


def _superseded_sql(table: str) -> str:
    match = " AND ".join(f"n.{k} = t.{k}" for k in PRUNE_KEYS[table])
    return (
        f"FROM {table} t WHERE t.season = %s AND EXISTS ("
        f"SELECT 1 FROM {table} n WHERE {match} AND n.recorded_on > t.recorded_on)"
    )


def read_season(conn, table: str, season: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table} WHERE season = %s ORDER BY recorded_on", (season,))
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def count_superseded(conn, season: str) -> dict[str, tuple[int, int]]:
    """Per table: (rows of the season, rows that pruning would delete)."""
    out = {}
    with conn.cursor() as cur:
        for table in PRUNE_KEYS:
            cur.execute(f"SELECT count(*) FROM {table} WHERE season = %s", (season,))
            total = cur.fetchone()[0]
            cur.execute(f"SELECT count(*) {_superseded_sql(table)}", (season,))
            out[table] = (total, cur.fetchone()[0])
    return out


def prune_season(conn, season: str, current_season: str) -> dict[str, int]:
    """Delete, for a FINISHED season, every prediction superseded by a later one of the same
    game (and model). Evaluation results do not change. Refuses the current or a later season.
    Does NOT commit: the caller checks the result, then commits or rolls back.
    """
    if season >= current_season:
        raise ValueError(f"{season} is not finished (current season is {current_season}).")
    deleted = {}
    with conn.cursor() as cur:
        for table in PRUNE_KEYS:
            cur.execute(f"DELETE {_superseded_sql(table)}", (season,))
            deleted[table] = cur.rowcount
    return deleted
