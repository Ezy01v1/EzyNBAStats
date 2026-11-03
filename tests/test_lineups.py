import numpy as np
import pandas as pd
import pytest

from sportslab.analysis.lineups import (
    add_uncertainty,
    margin_sd_per_game,
    net_rating_se,
    paired_error_gain,
    paired_halves,
    persistence,
    prediction_errors,
    shrink,
)
from sportslab.data.lineups import clean_lineups, clean_team_ratings


def _raw_lineups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GROUP_ID": ["-1-2-", "-3-4-"],
            "GROUP_NAME": ["A - B", "C - D"],
            "TEAM_ID": [10, 20],
            "TEAM_ABBREVIATION": ["BOS", "NYK"],
            "GP": [30, 20],
            "MIN": [300.0, 150.0],
            "POSS": [650, 330],
            "OFF_RATING": [115.0, 110.0],
            "DEF_RATING": [108.0, 112.0],
            "NET_RATING": [7.0, -2.0],
            "EXTRA": [0, 0],
        }
    )


def test_clean_lineups_renames_and_rejects_duplicate_ids():
    out = clean_lineups(_raw_lineups())
    assert {"lineup_id", "players", "team_id", "minutes", "poss", "net_rating"} <= set(out.columns)
    assert "EXTRA" not in out.columns
    raw = _raw_lineups()
    raw.loc[1, "GROUP_ID"] = "-1-2-"
    raw.loc[1, "TEAM_ID"] = 10  # same lineup under the same team: a real duplicate
    with pytest.raises(ValueError, match="one row per lineup id and team"):
        clean_lineups(raw)
    raw.loc[1, "TEAM_ID"] = 99  # same players under another team (a trade) is legitimate
    assert len(clean_lineups(raw)) == 2


def test_clean_team_ratings_keeps_ids_and_net_rating():
    raw = pd.DataFrame(
        {
            "TEAM_ID": [1],
            "TEAM_NAME": ["Boston Celtics"],
            "GP": [40],
            "POSS": [3900],
            "NET_RATING": [7.1],
            "EXTRA": [0],
        }
    )
    assert list(clean_team_ratings(raw).columns) == ["team_id", "team", "gp", "poss", "net_rating"]


def test_margin_sd_per_game_uses_final_margins():
    games = pd.DataFrame({"home_pts": [110, 100, 120, 90], "away_pts": [100, 100, 100, 100]})
    assert margin_sd_per_game(games) == pytest.approx(np.std([10, 0, 20, -10], ddof=1))


def test_standard_error_falls_with_the_square_root_of_possessions():
    se = net_rating_se(np.array([100, 400, 1600]), margin_sd=16.0)
    assert se.tolist() == pytest.approx([16.0, 8.0, 4.0])


def test_uncertainty_columns_bracket_the_rating():
    out = add_uncertainty(clean_lineups(_raw_lineups()), margin_sd=16.0)
    assert (out["ci_low"] < out["net_rating"]).all() and (out["net_rating"] < out["ci_high"]).all()
    assert out.loc[1, "se"] > out.loc[0, "se"]  # fewer possessions, wider interval


def test_shrink_is_the_possession_weighted_average():
    assert shrink(pd.Series([10.0]), pd.Series([300.0]), 0.0, 100.0).iloc[0] == pytest.approx(7.5)
    assert shrink(pd.Series([10.0]), pd.Series([300.0]), 2.0, 0.0).iloc[0] == pytest.approx(10.0)


def _simulate(true_sd: float, n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_teams = 20
    team_effect = rng.normal(0, 4, n_teams)
    team_id = rng.integers(0, n_teams, n)
    lineup_effect = rng.normal(0, true_sd, n)
    true_net = team_effect[team_id] + lineup_effect
    poss = rng.integers(300, 1200, size=(2, n))
    noise_sd = 16.0 * np.sqrt(100 / poss)
    nets = true_net + rng.normal(size=(2, n)) * noise_sd
    cols = ["lineup_id", "players", "team_id", "net_rating", "poss"]

    def half(i):
        return pd.DataFrame(
            {
                "lineup_id": range(n),
                "players": "x",
                "team_id": team_id,
                "net_rating": nets[i],
                "poss": poss[i],
            }
        )[cols]

    teams = pd.DataFrame({"team_id": range(n_teams), "net_rating": team_effect})
    return paired_halves(half(0), half(1), teams, min_poss=300)


def test_paired_halves_filters_by_possessions_in_both_halves_and_adds_team_rating():
    joined = _simulate(true_sd=3)
    assert {"net_1", "net_2", "poss_1", "poss_2", "team_net_1"} <= set(joined.columns)
    assert (joined["poss_1"] >= 300).all() and (joined["poss_2"] >= 300).all()
    strict = paired_halves(
        pd.DataFrame(
            {"lineup_id": [1], "players": "x", "team_id": [0], "net_rating": [1.0], "poss": [200]}
        ),
        pd.DataFrame({"lineup_id": [1], "team_id": [0], "net_rating": [1.0], "poss": [900]}),
        pd.DataFrame({"team_id": [0], "net_rating": [2.0]}),
        min_poss=300,
    )
    assert strict.empty


def test_persistence_is_high_when_lineups_truly_differ_and_low_beyond_team_when_they_do_not():
    differ = persistence(_simulate(true_sd=8), n_boot=100)
    assert differ["raw"][1] > 0 and differ["beyond_team"][1] > 0  # intervals' lower ends above 0
    same = persistence(_simulate(true_sd=0, n=600), n_boot=100)
    assert same["raw"][1] > 0  # team strength alone makes ratings persist (interval above 0)...
    assert same["beyond_team"][1] < 0.2  # ...but beyond the team there is nothing to persist


def test_shrinking_towards_the_team_beats_raw_when_lineups_do_not_differ():
    joined = _simulate(true_sd=0, n=600)
    errors = prediction_errors(joined, ks=[0, 200, 1_000_000], prior="team")
    assert errors[1_000_000] < errors[0] and errors[200] < errors[0]


def test_raw_rating_beats_team_only_when_lineups_truly_differ():
    joined = _simulate(true_sd=12, n=800)
    errors = prediction_errors(joined, ks=[0, 1_000_000], prior="team")
    assert errors[0] < errors[1_000_000]


def test_paired_error_gain_is_positive_with_an_interval_above_zero_when_shrinking_clearly_helps():
    joined = _simulate(true_sd=0, n=600)
    gain, low, high = paired_error_gain(joined, k_ref=0, k_new=1_000_000, n_boot=200)
    assert low > 0 and low <= gain <= high


def test_paired_error_gain_of_a_predictor_against_itself_is_exactly_zero():
    gain, low, high = paired_error_gain(_simulate(true_sd=3), 200, 200, n_boot=50)
    assert gain == pytest.approx(0) and low == pytest.approx(0) and high == pytest.approx(0)
