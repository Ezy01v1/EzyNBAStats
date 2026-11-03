import pandas as pd
import pytest

from sportslab.features.team_form import FORM_PRIOR_GAMES, MAX_REST_DAYS, build_pregame_features
from sportslab.models.elo import EloParams


def _games(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, season, home_id, away_id, home_pts, away_pts), chronological."""
    df = pd.DataFrame(
        rows,
        columns=["game_date", "season", "home_team_id", "away_team_id", "home_pts", "away_pts"],
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_id"] = [f"{i:03d}" for i in range(len(df))]
    df["home_team"] = "H" + df["home_team_id"].astype(str)
    df["away_team"] = "A" + df["away_team_id"].astype(str)
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    df["neutral_site"] = False
    return df


def _build(rows) -> pd.DataFrame:
    return build_pregame_features(_games(rows), EloParams())


def test_rest_days_and_back_to_back():
    # Team 1 plays on Jan 1, Jan 2 (back-to-back), Jan 5 (3 days rest).
    out = _build(
        [
            ("2024-01-01", "2023-24", 1, 2, 100, 90),
            ("2024-01-02", "2023-24", 1, 3, 100, 90),
            ("2024-01-05", "2023-24", 1, 4, 100, 90),
        ]
    )
    assert list(out["home_rest"]) == [MAX_REST_DAYS, 1, 3]
    assert list(out["home_b2b"]) == [0, 1, 0]
    assert list(out["away_b2b"]) == [0, 0, 0]


def test_rest_is_capped_so_the_offseason_is_not_a_huge_number():
    out = _build(
        [
            ("2024-04-01", "2023-24", 1, 2, 100, 90),
            ("2024-10-25", "2024-25", 1, 2, 100, 90),
        ]
    )
    assert out["home_rest"].iloc[1] == MAX_REST_DAYS


def test_form_is_shrunk_toward_zero_with_few_games():
    # Team 1 wins by 10 then by 20. Before game 3: 30 / (2 games + prior) -- not the raw 15.
    out = _build(
        [
            ("2024-01-01", "2023-24", 1, 2, 110, 100),
            ("2024-01-03", "2023-24", 3, 1, 90, 110),  # team 1 away, wins by 20
            ("2024-01-05", "2023-24", 1, 4, 100, 100 - 1),
        ]
    )
    assert out["home_form"].iloc[0] == 0.0  # no history yet
    assert out["home_form"].iloc[2] == pytest.approx(30 / (2 + FORM_PRIOR_GAMES))


def test_form_resets_each_season():
    out = _build(
        [
            ("2024-01-01", "2023-24", 1, 2, 150, 100),
            ("2024-10-25", "2024-25", 1, 2, 100, 100 - 1),
        ]
    )
    assert out["home_form"].iloc[1] == 0.0


def test_form_only_looks_at_the_last_ten_games():
    rows = [(f"2024-01-{d:02d}", "2023-24", 1, 2, 200, 100) for d in range(1, 3)]  # two +100s
    rows += [(f"2024-02-{d:02d}", "2023-24", 1, 2, 101, 100) for d in range(1, 11)]  # ten +1s
    rows += [("2024-03-01", "2023-24", 1, 2, 101, 100)]
    out = _build(rows)
    # window = the ten +1 games only (the +100s fell out): 10 / (10 + prior)
    assert out["home_form"].iloc[-1] == pytest.approx(10 / (10 + FORM_PRIOR_GAMES))


def test_diff_features_are_home_minus_away():
    out = _build(
        [
            ("2024-01-01", "2023-24", 1, 2, 110, 100),
            ("2024-01-02", "2023-24", 3, 1, 100, 90),
            ("2024-01-05", "2023-24", 1, 2, 100, 99),
        ]
    )
    last = out.iloc[-1]
    assert last["form_diff"] == pytest.approx(last["home_form"] - last["away_form"])
    assert last["rest_diff"] == last["home_rest"] - last["away_rest"]
    assert last["elo_diff"] == pytest.approx(last["elo_home_pre"] - last["elo_away_pre"])


def test_no_leakage_a_games_own_result_never_changes_its_features():
    base = [
        ("2024-01-01", "2023-24", 1, 2, 110, 100),
        ("2024-01-03", "2023-24", 2, 3, 95, 100),
        ("2024-01-05", "2023-24", 1, 3, 100, 90),
    ]
    changed = [*base[:2], ("2024-01-05", "2023-24", 1, 3, 60, 130)]  # rewrite only the LAST game
    a, b = _build(base), _build(changed)
    feature_cols = [
        "elo_diff",
        "form_diff",
        "rest_diff",
        "home_b2b",
        "away_b2b",
        "home_form",
        "away_form",
    ]
    pd.testing.assert_frame_equal(a[feature_cols], b[feature_cols])


def test_features_have_no_missing_values():
    out = _build(
        [
            ("2024-01-01", "2023-24", 1, 2, 110, 100),
            ("2024-01-02", "2023-24", 3, 4, 95, 100),
        ]
    )
    assert (
        not out[["elo_diff", "form_diff", "rest_diff", "home_b2b", "away_b2b"]].isna().any().any()
    )
