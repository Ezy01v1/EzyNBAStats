import pandas as pd
import pytest

from sportslab.models.elo import EloParams, expected_home_win, run_elo


def _games(rows: list[tuple[str, int, int, int]]) -> pd.DataFrame:
    """rows: (season, home_team_id, away_team_id, home_win), already in chronological order."""
    df = pd.DataFrame(rows, columns=["season", "home_team_id", "away_team_id", "home_win"])
    df["game_date"] = pd.date_range("2024-01-01", periods=len(df))
    return df


def test_equal_ratings_without_home_advantage_is_a_coin_flip():
    assert expected_home_win(1500, 1500, 0) == pytest.approx(0.5)


def test_400_points_is_ten_to_one_odds():
    assert expected_home_win(1900, 1500, 0) == pytest.approx(10 / 11)


def test_neutral_site_game_gets_no_home_advantage():
    games = _games([("2023-24", 1, 2, 1)])
    games["neutral_site"] = True
    out = run_elo(games, EloParams(home_advantage=100))
    assert out["p_home"].iloc[0] == pytest.approx(0.5)


def test_home_advantage_favors_home_team():
    assert expected_home_win(1500, 1500, 100) > 0.5


def test_first_game_prediction_uses_initial_ratings():
    out = run_elo(_games([("2023-24", 1, 2, 1)]), EloParams(home_advantage=100))
    assert out["p_home"].iloc[0] == pytest.approx(expected_home_win(1500, 1500, 100))


def test_update_is_zero_sum_and_moves_toward_the_result():
    params = EloParams(k=20, home_advantage=0)
    out = run_elo(_games([("2023-24", 1, 2, 1), ("2023-24", 1, 2, 1)]), params)
    # after game 1 (home win, p=.5): home +10, away -10
    assert out["elo_home_pre"].iloc[1] == pytest.approx(1510)
    assert out["elo_away_pre"].iloc[1] == pytest.approx(1490)


def test_upset_moves_ratings_more_than_expected_win():
    params = EloParams(k=20, home_advantage=0)
    # Team 1 wins big first, then loses at home: that loss should cost more than 10 points.
    rows = [("2023-24", 1, 2, 1), ("2023-24", 1, 2, 1), ("2023-24", 1, 2, 0), ("2023-24", 1, 2, 1)]
    pre = run_elo(_games(rows), params)["elo_home_pre"]
    # Team 1 was favored (p > .5) in game 3 and lost: it drops more than the 10 points an
    # even game would cost.
    assert pre.iloc[2] - pre.iloc[3] > 10


def test_season_boundary_regresses_ratings_to_the_mean():
    params = EloParams(k=20, home_advantage=0, carryover=0.5)
    out = run_elo(_games([("2022-23", 1, 2, 1), ("2023-24", 1, 2, 1)]), params)
    # end of 22-23: team 1 = 1510. Carryover 0.5 -> 1500 + 0.5*10 = 1505
    assert out["elo_home_pre"].iloc[1] == pytest.approx(1505)
    assert out["elo_away_pre"].iloc[1] == pytest.approx(1495)


def test_no_lookahead_prediction_ignores_the_game_result_itself_and_the_future():
    base = [("2023-24", 1, 2, 1), ("2023-24", 2, 3, 0), ("2023-24", 3, 1, 1)]
    flipped = [*base[:2], ("2023-24", 3, 1, 0)]  # change only the LAST result
    a = run_elo(_games(base))["p_home"].tolist()
    b = run_elo(_games(flipped))["p_home"].tolist()
    assert a == b  # a game's own result can't change its own prediction


def test_probabilities_stay_in_open_unit_interval():
    out = run_elo(_games([("2023-24", 1, 2, 1)] * 200), EloParams(k=40))
    assert out["p_home"].between(0, 1, inclusive="neither").all()
