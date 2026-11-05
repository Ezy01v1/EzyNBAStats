"""End-to-end check of the predictions log by replaying a finished season day by day.

Run: ``python -m uv run python scripts/backtest_predictions_log.py``

For every day of 2025-26 it rebuilds the Elo from the games before that day only, records the
day's predictions with the same functions as the daily job, writes them to Neon (a throwaway
schema, dropped at the end), scores them with the real results and evaluates them. Fantasy is
replayed on a sample of days (refitting the model each time is slow). If the pipeline is right,
the numbers must match the offline validations of Modules 1 and 10 on the same season.
"""

import time
import uuid

import pandas as pd
import psycopg

from sportslab.data import NbaClient
from sportslab.data.gamelog import load_player_games
from sportslab.data.games import load_games
from sportslab.data.schedule import load_conferences, load_schedule
from sportslab.features.team_form import build_pregame_features
from sportslab.models.elo import run_elo
from sportslab.models.fantasy_projection import project_next_games
from sportslab.models.live_logistic import (
    LOGISTIC_MODEL,
    fit_logistic,
    pending_features,
    predict_pending,
)
from sportslab.models.season_state import build_season_state
from sportslab.service.context import TUNED_ELO, history_seasons
from sportslab.storage import predictions as store

SEASON = "2025-26"
TRAIN = ["2021-22", "2022-23", "2023-24"]  # nothing from the replayed season
CALIBRATION = "2024-25"
FANTASY_DAYS = 12

t0 = time.time()
client = NbaClient()
history = load_games(client, history_seasons(SEASON))
schedule = load_schedule(client, SEASON)
conferences = load_conferences(client, SEASON)
players = load_player_games(client, TRAIN + [CALIBRATION, SEASON])
schedule = schedule[schedule["played"]].reset_index(drop=True)  # the finished season
days = sorted(schedule["game_date"].dt.normalize().unique())
print(f"{SEASON}: {len(schedule)} games on {len(days)} days")

logistic = fit_logistic(build_pregame_features(history, TUNED_ELO))  # before the season only
game_rows = []
for day in days:
    as_of = schedule.assign(played=schedule["game_date"] < day)  # what was known that morning
    state = build_season_state(history, as_of, conferences, SEASON, TUNED_ELO)
    rows = store.game_prediction_rows(
        as_of, state.ratings, TUNED_ELO, SEASON, day.date(), now=day.tz_localize("UTC")
    )
    logit = store.model_game_rows(
        predict_pending(logistic, pending_features(as_of, SEASON, state.ratings)),
        LOGISTIC_MODEL,
        SEASON,
        day.date(),
        now=day.tz_localize("UTC"),
    )
    rows = pd.concat([rows, logit], ignore_index=True)
    game_rows.append(rows[rows["game_date"] == day.date()])  # that day's games
games = pd.concat(game_rows, ignore_index=True)

fantasy_rows = []
step = max(len(days) // FANTASY_DAYS, 1)
for day in days[10::step][:FANTASY_DAYS]:
    known = players[players["game_date"] < day]
    as_of = schedule.assign(played=schedule["game_date"] < day)
    proj = project_next_games(known, as_of, TRAIN, CALIBRATION)
    rows = store.fantasy_prediction_rows(proj, SEASON, day.date())
    fantasy_rows.append(rows[rows["game_date"] == day.date()])
fantasy = pd.concat(fantasy_rows, ignore_index=True)
print(f"Built {len(games)} game and {len(fantasy)} fantasy predictions ({time.time() - t0:.0f}s)")

direct = store.database_url().replace("-pooler", "")
schema = f"backtest_{uuid.uuid4().hex[:8]}"
with psycopg.connect(direct, autocommit=True) as admin:
    admin.execute(f"CREATE SCHEMA {schema}")
try:
    with psycopg.connect(direct, options=f"-c search_path={schema}") as conn:
        store.ensure_schema(conn)
        n_games = store.record_game_predictions(conn, games)
        n_fantasy = store.record_fantasy_predictions(conn, fantasy)
        assert store.record_game_predictions(conn, games) == 0, "re-recording added rows"
        scored_games = store.score_games(conn, schedule)
        scored_fantasy = store.score_fantasy(conn, players[players["season"] == SEASON])
        stored_games = store.read_game_predictions(conn)
        stored_fantasy = store.read_fantasy_predictions(conn)
finally:
    with psycopg.connect(direct, autocommit=True) as admin:
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
print(
    f"Neon round trip: recorded {n_games} / {n_fantasy}, scored {scored_games} / "
    f"{scored_fantasy} ({time.time() - t0:.0f}s); schema {schema} dropped"
)

elo_name = store.elo_label(TUNED_ELO)
g = store.evaluate_games(stored_games[stored_games["model"] == elo_name])
reference = run_elo(
    pd.concat(
        [history, load_games(client, [SEASON])], ignore_index=True
    ),  # Module 1's offline walk
    TUNED_ELO,
)
reference = reference[reference["season"] == SEASON]
lo, hi = g["favorite_won_95"]
print(
    f"\nGames (n={g['n']}): Brier {g['brier']:.4f}, log loss {g['log_loss']:.4f}, "
    f"favorite won {g['favorite_won']:.1%} [{lo:.1%}, {hi:.1%}]"
)
elo_rows = stored_games[stored_games["model"] == elo_name]
merged = elo_rows.merge(reference[["game_id", "p_home"]], on="game_id", suffixes=("", "_ref"))
diff = (merged["p_home"] - merged["p_home_ref"]).abs()
print(
    f"vs the offline Elo walk on the same {len(merged)} games: mean |diff p| {diff.mean():.4f}, "
    f"max {diff.max():.4f} (only same-day games differ: the walk updates within a day)"
)
print(g["calibration"].to_string(index=False))
lg = store.evaluate_games(stored_games[stored_games["model"] == LOGISTIC_MODEL])
c = store.compare_models(stored_games, LOGISTIC_MODEL, elo_name)
lo, hi = c["gain_95"]
print(
    f"Logistic (n={lg['n']}, no neutral sites): Brier {lg['brier']:.4f}, "
    f"log loss {lg['log_loss']:.4f}; gain over Elo on the same {c['n']} games "
    f"{c['gain']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
)

f = store.evaluate_fantasy(stored_fantasy)
print(
    f"\nFantasy (n={f['n']} player-games on {FANTASY_DAYS} days, {f['players']} players): "
    f"MAE {f['mae']:.2f}, bias {f['bias']:+.2f}"
)
for level in (50, 80):
    lo, hi = f[f"coverage{level}_95"]
    print(f"  {level}% interval covered {f[f'coverage{level}']:.1%} [{lo:.1%}, {hi:.1%}]")
unscored = stored_fantasy["actual"].isna().mean()
print(f"  projections without a game played (did not play): {unscored:.1%}")
