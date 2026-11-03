import pandas as pd
import pytest

from sportslab.analysis.daily import (
    add_probabilities,
    games_on,
    last_game_day,
    next_game_day,
    standings,
    surprises,
    top_performances,
)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "game_date": pd.to_datetime(["2026-11-01", "2026-11-02", "2026-11-02", "2026-11-04"]),
            "played": [True, True, True, False],
            "home_team_id": [1, 2, 3, 1],
            "away_team_id": [2, 3, 1, 3],
            "home_pts": [100, 90, 110, 0],
            "away_pts": [90, 95, 100, 0],
            "neutral_site": [False, False, False, False],
        }
    )


def test_day_helpers_find_the_last_played_day_and_the_next_with_games():
    s = _schedule()
    assert last_game_day(s) == pd.Timestamp("2026-11-02")
    assert next_game_day(s, pd.Timestamp("2026-11-03")) == pd.Timestamp("2026-11-04")
    assert next_game_day(s, pd.Timestamp("2026-11-05")) is None
    assert len(games_on(s, pd.Timestamp("2026-11-02 18:00"))) == 2
    assert last_game_day(s.assign(played=False)) is None


def test_probabilities_favor_the_stronger_home_team_and_drop_the_bonus_when_neutral():
    s = _schedule().iloc[:1]
    strong = add_probabilities(s, {1: 1600, 2: 1500}, 40)
    weak = add_probabilities(s, {1: 1500, 2: 1600}, 40)
    assert strong.loc[0, "p_home"] > 0.5 > weak.loc[0, "p_home"]
    even = add_probabilities(s.assign(neutral_site=True), {1: 1500, 2: 1500}, 40)
    assert even.loc[0, "p_home"] == pytest.approx(0.5)
    default_rated = add_probabilities(s, {}, 0)
    assert default_rated.loc[0, "p_home"] == pytest.approx(0.5)


def test_surprises_are_wins_by_sides_that_had_a_low_probability():
    s = _schedule().iloc[:3].copy()
    s["p_home"] = [0.9, 0.8, 0.30]  # a: expected home win; b: home (80%) lost; c: home (30%) won
    out = surprises(s, threshold=0.35)
    assert out["game_id"].tolist() == ["b", "c"]
    assert out["p_winner"].round(2).tolist() == [0.2, 0.3]


def test_standings_count_wins_form_and_list_teams_without_games():
    conf = pd.DataFrame(
        {
            "team_id": [1, 2, 3, 4],
            "team": list("ABCD"),
            "conference": ["East", "East", "West", "West"],
        }
    )
    table = standings(_schedule().iloc[:3], conf, last_n=2)
    a = table[table["team"] == "A"].iloc[0]  # beat B 100-90 at home, lost 100-110 away to C
    assert (a["wins"], a["losses"], a["games"]) == (1, 1, 2) and a["last_n"] == "1-1"
    c = table[table["team"] == "C"].iloc[0]  # beat B? no: B-C: C won away 95-90; then beat A
    assert (c["wins"], c["losses"]) == (2, 0)
    d = table[table["team"] == "D"].iloc[0]
    assert d["games"] == 0 and d["win_pct"] == 0.0
    west = table[table["conference"] == "West"].reset_index(drop=True)
    assert west.loc[0, "team"] == "C"  # ranked inside its own conference


def test_top_performances_lists_the_best_fantasy_games_of_the_day():
    games = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-11-02"] * 3 + ["2026-11-01"]),
            "player": list("ABCD"),
            "team": "AAA",
            "opponent": "BBB",
            "minutes": 30.0,
            "pts": 10,
            "reb": 5,
            "ast": 5,
            "stl": 1,
            "blk": 0,
            "tov": 2,
            "fantasy": [30.0, 50.0, 40.0, 99.0],
        }
    )
    out = top_performances(games, pd.Timestamp("2026-11-02"), n=2)
    assert out["player"].tolist() == ["B", "C"]  # D belongs to another day
