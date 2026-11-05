"""Shrink a FINISHED season in the predictions log, keeping what the evaluation uses.

Run (dry run, deletes nothing):  ``python -m uv run python scripts/prune_season.py 2026-27``
Delete for real:                 ``python -m uv run python scripts/prune_season.py 2026-27 --apply``

Every day stores a prediction for every remaining game, so a game ends up with ~100 copies; only
the latest one before tip-off is evaluated. With ``--apply`` the whole season is first exported to
``data/processed/predictions_<season>_<table>.parquet`` (git-ignored), the export is read back and
checked, and only then are the superseded rows deleted, in one transaction that is committed only
if the evaluation numbers are identical before and after. The current season is refused.
"""

import argparse
from pathlib import Path

import pandas as pd

from sportslab.service.context import CURRENT
from sportslab.storage import predictions as store

BACKUP_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

parser = argparse.ArgumentParser()
parser.add_argument("season", help="finished season, e.g. 2026-27")
parser.add_argument("--apply", action="store_true", help="export, then delete for real")
args = parser.parse_args()
if args.season >= CURRENT:
    raise SystemExit(f"{args.season} is not finished (current season is {CURRENT}).")


def summary(conn) -> dict:
    games = store.read_season(conn, "game_predictions", args.season)
    fantasy = store.read_season(conn, "fantasy_predictions", args.season)
    by_model = store.evaluate_games_by_model(games) if len(games) else {}
    out = {m: (r["n"], round(r.get("log_loss", 0), 10)) for m, r in by_model.items()}
    f = store.evaluate_fantasy(fantasy) if len(fantasy) else {"n": 0}
    out["fantasy"] = (f["n"], round(f.get("mae", 0), 10))
    return out


with store.connect() as conn:
    counts = store.count_superseded(conn, args.season)
    for table, (total, old) in counts.items():
        print(
            f"{table}: {total} rows in {args.season}, {old} superseded (would keep {total - old})"
        )
    if not args.apply:
        raise SystemExit("Dry run: nothing deleted. Add --apply to export and delete.")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for table, (total, _) in counts.items():
        path = BACKUP_DIR / f"predictions_{args.season}_{table}.parquet"
        frame = store.read_season(conn, table, args.season)
        frame.to_parquet(path, index=False)
        if len(pd.read_parquet(path)) != total:
            raise SystemExit(f"Backup {path} does not match the database; nothing deleted.")
        print(f"Backed up {total} rows to {path}")

    before = summary(conn)
    deleted = store.prune_season(conn, args.season, CURRENT)
    after = summary(conn)
    print(f"Evaluation before {before}\nEvaluation after  {after}")
    if before != after:
        conn.rollback()
        raise SystemExit("The evaluation would change: rolled back, nothing deleted.")
    conn.commit()
    print(f"Deleted: {deleted}")
