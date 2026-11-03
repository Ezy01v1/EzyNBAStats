import numpy as np
import pandas as pd
import pytest

from sportslab.analysis.court import CORNER_THREE_X, THREE_POINT_RADIUS, court_lines
from sportslab.analysis.shots import (
    BASIC,
    BASIC_RANGE,
    half_season_counts,
    league_zone_table,
    selection_vs_making,
    shrinkage_error,
    zone_table,
)
from sportslab.data.shots import clean_league_zones, clean_shots
from sportslab.models.metrics import wilson_interval


def _raw_shots() -> pd.DataFrame:
    n = 4
    return pd.DataFrame(
        {
            "GAME_ID": ["1"] * n,
            "GAME_DATE": ["20251207"] * n,
            "PLAYER_ID": [7] * n,
            "PERIOD": [1] * n,
            "ACTION_TYPE": ["Jump Shot"] * n,
            "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal", "3PT Field Goal", "2PT Field Goal"],
            "SHOT_ZONE_BASIC": ["Restricted Area", "Left Corner 3", "Left Corner 3", "Mid-Range"],
            "SHOT_ZONE_AREA": ["Center(C)", "Left Side(L)", "Left Side(L)", "Center(C)"],
            "SHOT_ZONE_RANGE": ["Less Than 8 ft.", "24+ ft.", "24+ ft.", "16-24 ft."],
            "SHOT_DISTANCE": [1, 22, 22, 18],
            "LOC_X": [0, -230, -230, 5],
            "LOC_Y": [5, 5, 5, 180],
            "SHOT_ATTEMPTED_FLAG": [1] * n,
            "SHOT_MADE_FLAG": [1, 1, 0, 0],
            "EXTRA": [0] * n,
        }
    )


def _raw_league() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GRID_TYPE": ["League Averages"] * 5,
            "SHOT_ZONE_BASIC": [
                "Restricted Area",
                "Left Corner 3",
                "Mid-Range",
                "Mid-Range",
                "Backcourt",
            ],
            "SHOT_ZONE_AREA": ["Center(C)", "Left Side(L)", "Center(C)", "Left Side(L)", "Back"],
            "SHOT_ZONE_RANGE": ["Less Than 8 ft.", "24+ ft.", "16-24 ft.", "8-16 ft.", "Back"],
            "FGA": [1000, 400, 300, 300, 10],
            "FGM": [650, 160, 120, 130, 5],
            "FG_PCT": [0.65, 0.4, 0.4, 0.433, 0.5],
        }
    )


def test_clean_shots_adds_point_value_and_dates():
    shots = clean_shots(_raw_shots())
    assert shots["points"].tolist() == [2, 3, 3, 2]
    assert shots["game_date"].iloc[0] == pd.Timestamp("2025-12-07")
    assert "EXTRA" not in shots.columns and "made" in shots.columns


def test_clean_shots_rejects_non_attempts():
    raw = _raw_shots()
    raw.loc[0, "SHOT_ATTEMPTED_FLAG"] = 0
    with pytest.raises(ValueError, match="SHOT_ATTEMPTED_FLAG"):
        clean_shots(raw)


def test_league_table_points_per_shot_uses_three_for_threes_and_drops_backcourt():
    lg = league_zone_table(clean_league_zones(_raw_league()))
    assert "Backcourt" not in lg.index
    assert lg.loc["Left Corner 3", "pps"] == pytest.approx(0.4 * 3)
    assert lg.loc["Restricted Area", "pps"] == pytest.approx(0.65 * 2)
    assert lg.loc["Mid-Range", "fg_pct"] == pytest.approx(250 / 600)
    assert lg["share"].sum() == pytest.approx(1)


def test_zone_table_lists_every_league_zone_even_with_zero_shots():
    table = zone_table(clean_shots(_raw_shots()), clean_league_zones(_raw_league()))
    assert set(table.index) == {"Restricted Area", "Left Corner 3", "Mid-Range"}
    assert table["fga"].sum() == 4
    empty = zone_table(clean_shots(_raw_shots()).iloc[:1], clean_league_zones(_raw_league()))
    assert empty.loc["Mid-Range", "fga"] == 0 and np.isnan(empty.loc["Mid-Range", "fg_pct"])
    # no shots -> the shrunk estimate is exactly the league rate
    assert empty.loc["Mid-Range", "shrunk_fg_pct"] == pytest.approx(
        empty.loc["Mid-Range", "league_fg_pct"]
    )


def test_shrinkage_moves_small_samples_to_the_league_and_leaves_big_ones_alone():
    league = clean_league_zones(_raw_league())
    league_rate = 650 / 1000
    few = pd.DataFrame({"zone_basic": ["Restricted Area"] * 4, "made": [0] * 4, "points": [2] * 4})
    many = pd.DataFrame(
        {"zone_basic": ["Restricted Area"] * 2000, "made": [0] * 2000, "points": [2] * 2000}
    )
    t_few = zone_table(few, league).loc["Restricted Area"]
    t_many = zone_table(many, league).loc["Restricted Area"]
    assert t_few["fg_pct"] == 0 and t_many["fg_pct"] == 0
    assert t_few["shrunk_fg_pct"] > 0.9 * league_rate  # 4 misses barely move the prior
    assert t_many["shrunk_fg_pct"] < 0.05  # 2000 misses overwhelm a 100-shot prior


def test_interval_excludes_league_only_with_enough_shots():
    league = clean_league_zones(_raw_league())  # restricted area league rate 0.65

    def shots(n, made):
        return pd.DataFrame(
            {
                "zone_basic": ["Restricted Area"] * n,
                "made": [1] * made + [0] * (n - made),
                "points": [2] * n,
            }
        )

    small = zone_table(shots(10, 4), league).loc["Restricted Area"]  # 40% on 10 shots
    large = zone_table(shots(500, 200), league).loc["Restricted Area"]  # 40% on 500 shots
    assert not small["excludes_league"] and large["excludes_league"]


def test_finer_zones_use_the_same_keys_for_player_and_league():
    table = zone_table(
        clean_shots(_raw_shots()), clean_league_zones(_raw_league()), keys=BASIC_RANGE
    )
    assert ("Mid-Range", "16-24 ft.") in table.index
    assert table.loc[("Mid-Range", "16-24 ft."), "fga"] == 1


def test_selection_and_making_add_up_to_actual_minus_league():
    shots = clean_shots(_raw_shots())
    league = clean_league_zones(_raw_league())
    r = selection_vs_making(shots, league, BASIC)
    assert r["n"] == 4
    assert r["selection"] + r["making"] + r["league_pps"] == pytest.approx(r["actual_pps"])
    assert r["making_low"] <= r["making"] <= r["making_high"]


def test_wilson_interval_is_inside_0_1_and_wider_for_fewer_shots():
    low10, high10 = wilson_interval(0, 10)
    low1000, high1000 = wilson_interval(0, 1000)
    assert low10 == pytest.approx(0) and 0 < high10 < 1
    assert high1000 < high10
    low_n0, high_n0 = wilson_interval(0, 0)
    assert np.isnan(low_n0) and np.isnan(high_n0)


def test_three_point_arc_is_at_the_right_distance_and_corners_are_straight():
    lines = court_lines()
    x, y = lines["three_point_arc"]
    assert np.allclose(np.hypot(x, y), THREE_POINT_RADIUS)
    assert abs(x[0]) == pytest.approx(CORNER_THREE_X) and abs(x[-1]) == pytest.approx(
        CORNER_THREE_X
    )
    left_x, _ = lines["three_point_left"]
    assert (left_x == -CORNER_THREE_X).all()


def test_half_season_split_is_chronological_and_covers_every_shot():
    shots = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            "zone_basic": ["Mid-Range"] * 4,
            "made": [1, 1, 0, 0],
        }
    )
    c = half_season_counts(shots).loc["Mid-Range"]
    assert (c["fga_1"], c["fgm_1"], c["fga_2"], c["fgm_2"]) == (2, 2, 2, 0)  # early games first


def test_shrinkage_helps_when_rates_are_noisy_and_hurts_when_they_are_real():
    rng = np.random.default_rng(0)
    n_players = 400

    def counts(true_sd):
        rate = np.clip(0.4 + rng.normal(0, true_sd, n_players), 0.05, 0.95)
        fga = rng.integers(10, 40, size=(2, n_players))
        fgm = rng.binomial(fga, rate)
        return pd.DataFrame(
            {"fga_1": fga[0], "fgm_1": fgm[0], "fga_2": fga[1], "fgm_2": fgm[1], "league_rate": 0.4}
        )

    same_skill = counts(true_sd=0.0)  # everybody is the same: first-half rate is pure noise
    assert shrinkage_error(same_skill, 50) < shrinkage_error(same_skill, 0)
    real_skill = counts(true_sd=0.15)  # large true differences: shrinking too much loses signal
    assert shrinkage_error(real_skill, 5) < shrinkage_error(real_skill, 5000)
