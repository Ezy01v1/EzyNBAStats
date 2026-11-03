import numpy as np
import pandas as pd
import pytest

from sportslab.models.season_sim import (
    SimConfig,
    series_win_prob,
    simulate_season,
    win_prob,
)


def _league(strength: np.ndarray | None = None):
    ids = list(range(1, 31))
    conf = pd.DataFrame({"team_id": ids, "conference": ["East"] * 15 + ["West"] * 15, "team": ids})
    ratings = {t: 1500.0 + (0 if strength is None else strength[i]) for i, t in enumerate(ids)}
    return ids, conf, ratings


def _round_robin(ids, rounds: int = 2) -> pd.DataFrame:
    games = [
        (a, b) for _ in range(rounds) for a in ids for b in ids if a < b
    ]  # each pair meets `rounds` times at a's home
    return pd.DataFrame(
        {
            "home_team_id": [g[0] for g in games],
            "away_team_id": [g[1] for g in games],
            "neutral_site": False,
        }
    )


def test_win_prob_is_symmetric_without_home_advantage_and_monotonic_in_rating():
    assert win_prob(1500, 1500, 0) == pytest.approx(0.5)
    assert win_prob(1600, 1500, 0) + win_prob(1500, 1600, 0) == pytest.approx(1)
    assert win_prob(1500, 1500, 40) > 0.5 > win_prob(1500, 1500, -40)
    assert win_prob(1700, 1500, 0) > win_prob(1600, 1500, 0)


def test_series_probability_matches_known_best_of_seven_values():
    assert series_win_prob(0.5, 0.5) == pytest.approx(0.5)
    assert series_win_prob(1.0, 1.0) == pytest.approx(1.0)
    assert series_win_prob(0.0, 0.0) == pytest.approx(0.0)
    # classic result: a 60% per-game favorite wins a best-of-7 71.0% of the time
    assert series_win_prob(0.6, 0.6) == pytest.approx(0.7104, abs=1e-3)
    # being better at home only: strictly between the two constant cases
    mixed = series_win_prob(0.7, 0.5)
    assert series_win_prob(0.5, 0.5) < mixed < series_win_prob(0.7, 0.7)


def test_series_probability_works_on_arrays():
    out = series_win_prob(np.array([0.5, 0.6]), np.array([0.5, 0.6]))
    assert out == pytest.approx([0.5, 0.7104], abs=1e-3)


def test_stage_probabilities_add_up_to_the_number_of_slots():
    ids, conf, ratings = _league(np.linspace(-200, 200, 30))
    sim = simulate_season(
        ratings, conf, {t: 0 for t in ids}, _round_robin(ids), config=SimConfig(n_sims=2000)
    ).table
    assert sim["p_champion"].sum() == pytest.approx(1)
    assert sim["p_finals"].sum() == pytest.approx(2)
    assert sim["p_conf_finals"].sum() == pytest.approx(4)
    assert sim["p_round2"].sum() == pytest.approx(8)
    assert sim["p_playoffs"].sum() == pytest.approx(16)
    assert sim["p_top6"].sum() == pytest.approx(12)
    assert sim["p_playin"].sum() == pytest.approx(8)
    assert (sim["p_playoffs"] >= sim["p_top6"] - 1e-12).all()


def test_equal_teams_are_equally_likely_to_win_and_expected_wins_are_half_the_games():
    ids, conf, ratings = _league()
    games = _round_robin(ids)
    sim = simulate_season(
        ratings,
        conf,
        {t: 0 for t in ids},
        games.assign(neutral_site=True),
        config=SimConfig(n_sims=4000, home_advantage=0),
    ).table
    assert sim["exp_wins"].mean() == pytest.approx(len(games) / 30, abs=0.1)
    assert sim["p_champion"].max() < 0.08  # nobody close to a lock; 1/30 = 0.033 on average


def test_stronger_teams_win_more_and_current_wins_carry_over():
    strength = np.linspace(-300, 300, 30)
    ids, conf, ratings = _league(strength)
    current = {t: 0 for t in ids}
    current[1] = 30  # the weakest team starts 30 wins ahead
    sim = simulate_season(
        ratings, conf, current, _round_robin(ids), config=SimConfig(n_sims=2000)
    ).table
    assert sim.loc[30, "exp_wins"] > sim.loc[2, "exp_wins"]
    assert sim.loc[1, "exp_wins"] > 30 and sim.loc[1, "wins_p05"] >= 30
    assert sim["p_champion"].idxmax() in (29, 30)  # strongest of each conference side


def test_results_are_reproducible_for_a_seed_and_differ_between_seeds():
    ids, conf, ratings = _league(np.linspace(-150, 150, 30))
    args = (ratings, conf, {t: 0 for t in ids}, _round_robin(ids, 1))
    a = simulate_season(*args, config=SimConfig(n_sims=500, seed=3)).table
    b = simulate_season(*args, config=SimConfig(n_sims=500, seed=3)).table
    c = simulate_season(*args, config=SimConfig(n_sims=500, seed=4)).table
    pd.testing.assert_frame_equal(a, b)
    assert not a["p_champion"].equals(c["p_champion"])


def test_unknown_opponent_and_extra_games_add_expected_wins_without_ranking_the_unknown_team():
    ids, conf, ratings = _league()
    remaining = pd.DataFrame(
        {"home_team_id": [1, 2], "away_team_id": [0, 0], "neutral_site": [True, True]}
    )  # team 0 = "to be determined", not in the standings
    sim = simulate_season(
        ratings,
        conf,
        {t: 0 for t in ids},
        remaining,
        extra_games={1: 10},
        config=SimConfig(n_sims=3000, home_advantage=0),
    ).table
    assert 0 not in sim.index
    assert sim.loc[1, "exp_wins"] == pytest.approx(0.5 + 5.0, abs=0.15)  # 1 game + 10 extra at 50%
    assert sim.loc[2, "exp_wins"] == pytest.approx(0.5, abs=0.05)


def test_rating_uncertainty_widens_win_ranges_and_spreads_the_title_probability():
    strength = np.linspace(-250, 250, 30)
    ids, conf, ratings = _league(strength)
    args = (ratings, conf, {t: 0 for t in ids}, _round_robin(ids))
    tight = simulate_season(*args, config=SimConfig(n_sims=3000, rating_sd=0)).table
    loose = simulate_season(*args, config=SimConfig(n_sims=3000, rating_sd=80)).table
    width = lambda t: (t["wins_p95"] - t["wins_p05"]).mean()  # noqa: E731
    assert width(loose) > width(tight) + 2
    assert loose["p_champion"].max() < tight["p_champion"].max()  # the favorite is less of a lock
    assert loose["p_champion"].sum() == pytest.approx(1)
    assert loose["p_playoffs"].sum() == pytest.approx(16)
