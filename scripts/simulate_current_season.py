"""Simulate the rest of the season in progress and print the picture.

Run: ``python -m uv run python scripts/simulate_current_season.py``
"""

from datetime import timedelta

import pandas as pd

from sportslab.data import NbaClient
from sportslab.data.games import load_games, season_labels
from sportslab.data.schedule import load_conferences, load_schedule
from sportslab.models.elo import EloParams
from sportslab.models.season_sim import SimConfig, simulate_season
from sportslab.models.season_state import build_season_state

CURRENT = "2026-27"
TUNED_ELO = EloParams(k=24, home_advantage=40, carryover=0.7)

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 220

client = NbaClient()
history = load_games(client, season_labels(2015, 2025))
schedule = load_schedule(client, CURRENT, max_age=timedelta(hours=6))
conferences = load_conferences(client, CURRENT)
state = build_season_state(history, schedule, conferences, CURRENT, TUNED_ELO)
print(
    f"{CURRENT}: {state.games_played} games played up to {state.as_of.date()}; "
    f"{len(state.remaining)} scheduled; warnings: {state.warnings}"
)

sim = simulate_season(
    state.ratings,
    state.conferences,
    state.current_wins,
    state.remaining,
    state.extra_games,
    SimConfig(n_sims=10_000),
)
table = sim.table.join(conferences.set_index("team_id")[["team", "conference"]])
cols = [
    "team",
    "conference",
    "elo",
    "current_wins",
    "exp_wins",
    "wins_p10",
    "wins_p90",
    "p_playoffs",
    "p_finals",
    "p_champion",
]
print(table.sort_values("p_champion", ascending=False)[cols].head(12).to_string())
print(
    "sum title:",
    table["p_champion"].sum(),
    " widest p80 range:",
    (table["wins_p90"] - table["wins_p10"]).max(),
)
