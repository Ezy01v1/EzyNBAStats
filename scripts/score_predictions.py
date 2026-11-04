"""Fill in the results of recorded predictions and print how good they were.

Run: ``python -m uv run python scripts/score_predictions.py``
"""

import pandas as pd

from sportslab.data import NbaClient
from sportslab.data.gamelog import load_player_games
from sportslab.data.schedule import load_schedule
from sportslab.service.context import CURRENT, FRESHNESS
from sportslab.storage.predictions import (
    connect,
    ensure_schema,
    evaluate_fantasy,
    evaluate_games,
    read_fantasy_predictions,
    read_game_predictions,
    score_fantasy,
    score_games,
)

pd.options.display.float_format = "{:.3f}".format

client = NbaClient()
schedule = load_schedule(client, CURRENT, max_age=FRESHNESS)
players = load_player_games(client, [CURRENT], max_age=FRESHNESS)

with connect() as conn:
    ensure_schema(conn)
    scored_games = score_games(conn, schedule)
    scored_players = score_fantasy(conn, players)
    games = read_game_predictions(conn)
    fantasy = read_fantasy_predictions(conn)

print(f"Recorded: {len(games)} game predictions, {len(fantasy)} fantasy projections.")
print(f"With a result: {scored_games} games, {scored_players} player games.\n")

g = evaluate_games(games)
if g["n"]:
    lo, hi = g["favorite_won_95"]
    print(
        f"Games (n={g['n']}): Brier {g['brier']:.3f} (coin 0.250), log loss {g['log_loss']:.3f}, "
        f"favorite won {g['favorite_won']:.1%} (95% {lo:.1%}-{hi:.1%})"
    )
    print(g["calibration"].to_string(index=False))
else:
    print("Games: nothing scored yet.")

f = evaluate_fantasy(fantasy)
if f["n"]:
    print(
        f"\nFantasy (n={f['n']} player games, {f['players']} players): MAE {f['mae']:.2f}, "
        f"bias {f['bias']:+.2f}"
    )
    for level in (50, 80):
        lo, hi = f[f"coverage{level}_95"]
        print(f"  {level}% interval covered {f[f'coverage{level}']:.1%} (95% {lo:.1%}-{hi:.1%})")
else:
    print("Fantasy: nothing scored yet.")
