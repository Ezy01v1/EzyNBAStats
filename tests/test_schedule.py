import pandas as pd
import pytest

from sportslab.data.schedule import (
    clean_conferences,
    clean_schedule,
    games_per_team,
    playoff_outcomes,
)


def _raw_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gameId": ["0012600001", "0022600002", "0022600001", "0042600101"],
            "gameDate": ["10/01/2026 00:00:00", "10/21/2026 00:00:00", "10/20/2026 00:00:00"]
            + ["05/01/2027 00:00:00"],
            "gameStatus": [3, 1, 3, 1],
            "homeTeam_teamId": [1, 2, 1, 1],
            "awayTeam_teamId": [2, 1, 2, 2],
            "homeTeam_score": [100, 0, 110, 0],
            "awayTeam_score": [90, 0, 100, 0],
            "isNeutral": [False, False, True, False],
        }
    )


def test_schedule_keeps_only_regular_season_sorted_with_played_flag():
    out = clean_schedule(_raw_schedule())
    assert out["game_id"].tolist() == ["0022600001", "0022600002"]  # preseason/playoffs dropped
    assert out["played"].tolist() == [True, False]
    assert out["neutral_site"].tolist() == [True, False]
    assert out["game_date"].is_monotonic_increasing


def test_games_per_team_counts_home_and_away():
    counts = games_per_team(clean_schedule(_raw_schedule()))
    assert counts.to_dict() == {1: 2, 2: 2}


def test_duplicate_game_ids_are_rejected():
    raw = _raw_schedule()
    raw.loc[1, "gameId"] = "0022600001"
    with pytest.raises(ValueError, match="Duplicate"):
        clean_schedule(raw)


def test_conferences_are_renamed():
    raw = pd.DataFrame(
        {
            "TeamID": [1],
            "TeamName": ["Celtics"],
            "Conference": ["East"],
            "Division": ["Atlantic"],
            "WINS": [60],
        }
    )
    assert list(clean_conferences(raw).columns) == ["team_id", "team", "conference", "division"]


def _playoff_games() -> pd.DataFrame:
    rows = []
    # First round: 1 beats 2 (4-0). Finals: 1 beats 3 (4-2), the last series by date.
    for i in range(4):
        rows.append((f"a{i}", f"2027-04-{20 + i}", 1, 2, 1))
    for i in range(6):
        home, away = (1, 3) if i in (0, 1, 4) else (3, 1)
        home_win = int(i in (0, 1, 4, 3, 5)) if home == 1 else int(i in (2,))
        rows.append((f"b{i}", f"2027-06-{1 + i * 2:02d}", home, away, home_win))
    return pd.DataFrame(
        rows, columns=["game_id", "game_date", "home_team_id", "away_team_id", "home_win"]
    ).assign(game_date=lambda d: pd.to_datetime(d["game_date"]))


def test_playoff_outcomes_finds_participants_finalists_and_champion():
    games = _playoff_games()
    out = playoff_outcomes(games)
    assert out["made_playoffs"] == {1, 2, 3}
    assert out["finalists"] == {1, 3}
    assert out["champion"] == 1


def test_playoff_outcomes_of_an_empty_table_is_empty():
    out = playoff_outcomes(pd.DataFrame(columns=["game_id", "game_date", "home_team_id"]))
    assert out["champion"] is None and out["made_playoffs"] == set()


def test_played_rows_become_a_game_table_and_owed_games_are_the_gap_to_82():
    from sportslab.data.schedule import games_owed, schedule_to_games

    sch = clean_schedule(_raw_schedule())
    games = schedule_to_games(sch, "2026-27")
    assert len(games) == 1 and games.loc[0, "home_win"] == 1 and games.loc[0, "season"] == "2026-27"
    owed = games_owed(sch, pd.Series([1, 2, 3]), per_team=82)
    assert owed == {1: 80, 2: 80, 3: 82}
