import pandas as pd
import pytest

from sportslab.data.players import PLAYER_COLUMNS, RENAMES, clean_player_stats


def _raw(n: int = 3) -> pd.DataFrame:
    # Shape of LeagueDashPlayerStats Advanced: MIN is a per-game average, POSS a season total.
    raw = pd.DataFrame(
        {
            "PLAYER_ID": range(1, n + 1),
            "PLAYER_NAME": [f"P{i}" for i in range(1, n + 1)],
            "TEAM_ABBREVIATION": ["BOS", "NYK", "LAL"][:n],
            "AGE": [25, 30, 22][:n],
            "GP": [80, 40, 10][:n],
            "MIN": [30.0, 20.0, 5.0][:n],
            "TS_PCT": [0.60, 0.55, 0.50][:n],
            "USG_PCT": [0.25, 0.18, 0.15][:n],
            "AST_PCT": [0.20, 0.10, 0.05][:n],
            "REB_PCT": [0.10, 0.15, 0.08][:n],
            "OFF_RATING": [118.0, 112.0, 105.0][:n],
            "DEF_RATING": [110.0, 114.0, 108.0][:n],
            "NET_RATING": [8.0, -2.0, -3.0][:n],
            "PIE": [0.15, 0.09, 0.05][:n],
        }
    )
    raw["POSS"] = (raw["MIN"] * raw["GP"] * 2.1).round()  # ~2.1 possessions per minute
    raw["EXTRA_COLUMN_WE_IGNORE"] = 1
    return raw


def test_keeps_only_known_columns_with_clean_names():
    out = clean_player_stats(_raw(), "2025-26")
    assert list(out.columns) == PLAYER_COLUMNS
    assert set(out["season"]) == {"2025-26"}
    assert set(RENAMES.values()) <= set(out.columns)


def test_total_minutes_are_per_game_minutes_times_games():
    out = clean_player_stats(_raw(), "2025-26").set_index("player")
    assert out.loc["P1", "minutes"] == pytest.approx(2400)
    assert out.loc["P2", "minutes"] == pytest.approx(800)


def test_duplicate_players_are_rejected():
    raw = _raw()
    raw.loc[1, "PLAYER_ID"] = raw.loc[0, "PLAYER_ID"]
    with pytest.raises(ValueError, match="one row per player"):
        clean_player_stats(raw, "2025-26")


def test_minutes_semantics_change_is_caught():
    raw = _raw()
    raw["MIN"] = raw["MIN"] * raw["GP"]  # pretend NBA.com started returning season totals
    with pytest.raises(ValueError, match="possessions per minute"):
        clean_player_stats(raw, "2025-26")
