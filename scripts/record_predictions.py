"""Freeze today's predictions in the database (run once a day, before the games).

Run: ``python -m uv run python scripts/record_predictions.py``

Stores the Elo probability of every unplayed game and the fantasy projection (with 50% / 80%
intervals) of every active player's next game. Re-running on the same day changes nothing:
the first snapshot of a day wins. Games that already tipped off are skipped (not pre-game).
"""

from datetime import date

from sportslab.service.context import TUNED_ELO, DataService
from sportslab.storage.predictions import (
    connect,
    ensure_schema,
    fantasy_prediction_rows,
    game_prediction_rows,
    record_fantasy_predictions,
    record_game_predictions,
    started,
)

today = date.today()
service = DataService()
ctx, stale = service.context()
if stale:
    raise SystemExit("Data could not be refreshed; not recording stale predictions.")
games = game_prediction_rows(ctx.schedule, ctx.state.ratings, TUNED_ELO, ctx.season, today)
projections, stale = service.projections(ctx)
if stale:
    raise SystemExit("Player game logs could not be refreshed; not recording stale projections.")
in_progress = set(ctx.schedule.loc[started(ctx.schedule) & ~ctx.schedule["played"], "game_id"])
fantasy = fantasy_prediction_rows(projections, ctx.season, today, started_games=in_progress)

with connect() as conn:
    ensure_schema(conn)
    new_games = record_game_predictions(conn, games)
    new_fantasy = record_fantasy_predictions(conn, fantasy)

print(
    f"{today}: {new_games} of {len(games)} game predictions and "
    f"{new_fantasy} of {len(fantasy)} fantasy projections recorded (the rest already were)."
)
