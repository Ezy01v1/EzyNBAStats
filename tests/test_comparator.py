import numpy as np
import pandas as pd
import pytest

from sportslab.analysis.comparator import (
    binned_means,
    correlation_direction,
    eligible_pool,
    percentile_profile,
    rank_correlation,
    top_overlap,
)


def _players(n: int = 101) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": [f"P{i}" for i in range(n)],
            "minutes": np.linspace(0, 3000, n),
            "ts_pct": np.linspace(0.45, 0.65, n),
            "pie": np.linspace(0.0, 0.2, n),
        }
    )


def test_pool_keeps_only_players_with_enough_minutes():
    players = _players()
    pool = eligible_pool(players, min_minutes=1500)
    assert (pool["minutes"] >= 1500).all()
    assert len(pool) == int((players["minutes"] >= 1500).sum())


def test_percentile_is_midrank_and_monotonic():
    pool = _players(100)
    prof = percentile_profile(pool, pool, metrics=["ts_pct"])["ts_pct"]
    assert prof.is_monotonic_increasing
    assert prof.iloc[0] == pytest.approx(0.5) and prof.iloc[-1] == pytest.approx(99.5)


def test_percentile_of_ties_is_the_middle_of_their_block():
    pool = pd.DataFrame({"ts_pct": [0.5, 0.5, 0.5, 0.5]})
    assert percentile_profile(pool, pool, ["ts_pct"])["ts_pct"].eq(50).all()


def test_player_outside_pool_is_placed_against_the_pool():
    pool = pd.DataFrame({"ts_pct": [0.5, 0.6, 0.7, 0.8]})
    outsider = pd.DataFrame({"ts_pct": [0.1, 0.65, 0.9]})
    pct = percentile_profile(outsider, pool, ["ts_pct"])["ts_pct"].tolist()
    assert pct == [0.0, 50.0, 100.0]


def test_rank_correlation_detects_a_monotonic_relation_with_a_tight_interval():
    x = pd.Series(np.arange(200.0))
    rho, low, high = rank_correlation(x, x**3, n_boot=200)
    assert rho == pytest.approx(1.0) and low > 0.99 and high <= 1.0


def test_rank_correlation_without_association_straddles_zero():
    # A U shape has no monotonic relation: Spearman is exactly 0 by symmetry (and random noise
    # would exclude 0 in about 1 seed out of 20, making the test seed-dependent).
    x = pd.Series(np.arange(-75.0, 76.0))
    rho, low, high = rank_correlation(x, x**2, n_boot=300)
    assert rho == pytest.approx(0, abs=1e-9) and low < 0 < high


def test_partial_correlation_removes_what_a_confounder_explains():
    rng = np.random.default_rng(2)
    z = rng.normal(size=400)  # confounder (think: "is a big man")
    x = pd.Series(z + rng.normal(scale=0.3, size=400))
    y = pd.Series(z + rng.normal(scale=0.3, size=400))
    raw, *_ = rank_correlation(x, y, n_boot=50)
    partial, low, high = rank_correlation(x, y, control=pd.Series(z), n_boot=300)
    assert raw > 0.8
    assert low < 0 < high  # nothing left once z is accounted for


def test_correlation_direction_needs_the_whole_interval_on_one_side_of_zero():
    assert correlation_direction(0.05, 0.3) == "positive"
    assert correlation_direction(-0.3, -0.05) == "negative"
    assert correlation_direction(-0.1, 0.2) == "none"


def test_top_overlap_counts_shared_names():
    pool = pd.DataFrame(
        {"player": list("abcdef"), "u": [6, 5, 4, 3, 2, 1], "v": [6, 5, 1, 2, 3, 4]}
    )
    n, shared = top_overlap(pool, "u", "v", n=3)
    assert n == 2 and shared == {"a", "b"}


def test_binned_means_have_equal_counts_and_ordered_means():
    pool = _players(100)
    out = binned_means(pool, "ts_pct", "pie", bins=5)
    assert out["n"].tolist() == [20] * 5
    assert out["pie_mean"].is_monotonic_increasing
    assert (out["ci_low"] <= out["pie_mean"]).all() and (out["pie_mean"] <= out["ci_high"]).all()
