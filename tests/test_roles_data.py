import numpy as np
import pandas as pd
import pytest

from sportslab.data.profiles import ZONE_ATTEMPTS, clean_per100, clean_shot_zones
from sportslab.features.roles import FEATURES, build_role_features, standardize


def _zones_raw(as_strings: bool) -> pd.DataFrame:
    cols = [("", "PLAYER_ID"), ("", "PLAYER_NAME")]
    for zone in ZONE_ATTEMPTS:
        cols += [(zone, "FGM"), (zone, "FGA"), (zone, "FG_PCT")]
    rows = [
        [1, "A", *[1, 10, 0.1] * 5],
        [2, "B", *[2, 20, 0.1] * 5],
    ]
    frame = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(cols))
    if as_strings:  # what a Parquet round trip does to the column labels
        frame.columns = [str(c) for c in frame.columns]
    return frame


@pytest.mark.parametrize("as_strings", [False, True])
def test_shot_zones_parse_two_level_columns_in_both_forms(as_strings):
    out = clean_shot_zones(_zones_raw(as_strings))
    assert list(out["player_id"]) == [1, 2]
    assert list(out["ra_fga"]) == [10, 20] and list(out["atb3_fga"]) == [10, 20]
    assert set(ZONE_ATTEMPTS.values()) <= set(out.columns)


def test_per100_rejects_duplicate_players_and_renames():
    raw = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "FGA": [18.0, 12.0],
            "FG3A": [9.0, 2.0],
            "FTA": [5.0, 3.0],
            "OREB": [1.0, 4.0],
            "DREB": [5.0, 9.0],
            "AST": [6.0, 2.0],
            "TOV": [3.0, 2.0],
            "STL": [1.5, 1.0],
            "BLK": [0.5, 2.0],
            "PF": [3.0, 4.0],
            "MIN": [48.0, 48.0],
        }
    )
    out = clean_per100(raw)
    assert "min" not in out.columns and out["fga100"].tolist() == [18.0, 12.0]
    raw.loc[1, "PLAYER_ID"] = 1
    with pytest.raises(ValueError, match="one row per player"):
        clean_per100(raw)


def _inputs(n: int = 6):
    rng = np.random.default_rng(0)
    advanced = pd.DataFrame(
        {
            "season": "2025-26",
            "player_id": range(n),
            "player": [f"P{i}" for i in range(n)],
            "team": "AAA",
            "minutes": [2000.0, 2000.0, 500.0, 1500.0, 1200.0, 3000.0][:n],
            "usg_pct": rng.uniform(0.12, 0.3, n),
            "ts_pct": rng.uniform(0.5, 0.65, n),
        }
    )
    per100 = pd.DataFrame(
        {"player_id": range(n)}
        | {
            c: rng.uniform(1, 20, n)
            for c in [
                "fga100",
                "fg3a100",
                "fta100",
                "oreb100",
                "dreb100",
                "ast100",
                "tov100",
                "stl100",
                "blk100",
                "pf100",
            ]
        }
    )
    zones = pd.DataFrame(
        {"player_id": range(n)} | {c: rng.integers(5, 100, n) for c in ZONE_ATTEMPTS.values()}
    )
    return advanced, per100, zones


def test_role_features_keep_only_rotation_players_and_shares_sum_below_one():
    advanced, per100, zones = _inputs()
    out = build_role_features(advanced, per100, zones, min_minutes=1000)
    assert set(out["player"]) == {"P0", "P1", "P3", "P4", "P5"}  # P2 has 500 min
    assert set(FEATURES) <= set(out.columns)
    shares = out[["share_ra", "share_paint", "share_mid", "share_corner3"]].sum(axis=1)
    assert ((shares > 0) & (shares < 1)).all()


def test_standardize_gives_zero_mean_unit_std_and_reuses_a_reference():
    advanced, per100, zones = _inputs()
    out = build_role_features(advanced, per100, zones, min_minutes=0)
    z, mean, std = standardize(out)
    assert np.allclose(z.mean(), 0) and np.allclose(z.std(ddof=0), 1)
    shifted = out.copy()
    shifted["usg_pct"] += 0.1
    z2, _, _ = standardize(shifted, reference=out)
    assert (z2["usg_pct"] - z["usg_pct"]).round(6).gt(0).all()  # same ruler, so shift survives


def test_zone_attempts_missing_in_the_raw_table_mean_zero_attempts():
    raw = _zones_raw(as_strings=False)
    raw.loc[0, ("Mid-Range", "FGA")] = np.nan
    out = clean_shot_zones(raw)
    assert out.loc[0, "mid_fga"] == 0 and not out.isna().any().any()
