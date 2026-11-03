import pandas as pd

from sportslab.data.schedule import clean_schedule
from sportslab.models.elo import EloParams
from sportslab.models.season_state import build_season_state


def _schedule() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "gameId": ["0022600001", "0022600002", "0022600003", "0022600004"],
            "gameDate": ["10/20/2026 00:00:00", "10/21/2026 00:00:00", "10/25/2026 00:00:00"]
            + ["10/26/2026 00:00:00"],
            "gameStatus": [3, 3, 1, 1],
            "homeTeam_teamId": [1, 2, 1, 0],
            "awayTeam_teamId": [2, 1, 2, 2],
            "homeTeam_score": [100, 90, 0, 0],
            "awayTeam_score": [90, 95, 0, 0],
            "isNeutral": [False, False, False, False],
        }
    )
    return clean_schedule(raw)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2025-26"] * 2,
            "game_id": ["a", "b"],
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_win": [1, 0],
            "neutral_site": [False, False],
        }
    )


def test_state_splits_played_and_remaining_and_counts_wins():
    conf = pd.DataFrame({"team_id": [1, 2], "conference": ["East", "West"]})
    state = build_season_state(_history(), _schedule(), conf, "2026-27", EloParams())
    assert state.games_played == 2 and len(state.remaining) == 2
    assert state.current_wins == {1: 2, 2: 0}  # team 1 won both played games
    assert state.as_of == pd.Timestamp("2026-10-21")
    assert set(state.ratings) == {1, 2} and state.ratings[1] > state.ratings[2]
    assert 0 in set(state.remaining["home_team_id"])  # unknown opponent kept as "to be determined"


def test_incomplete_schedules_warn_and_owe_the_missing_games():
    conf = pd.DataFrame({"team_id": [1, 2], "conference": ["East", "West"]})
    state = build_season_state(_history(), _schedule(), conf, "2026-27", EloParams())
    assert state.extra_games == {1: 79, 2: 78}  # 3 and 4 games listed (one has a TBD side)
    assert state.warnings and "instead of 82" in state.warnings[0]
