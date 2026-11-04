import numpy as np
import pandas as pd
import pytest

from sportslab.data.gamelog import clean_gamelog, fantasy_points
from sportslab.features.fantasy import (
    append_next_games,
    build_features,
    eligible_rows,
    fill_missing,
)


def _raw(n: int = 3) -> pd.DataFrame:
    base = {
        "GAME_ID": [f"g{i}" for i in range(n)],
        "GAME_DATE": ["2025-11-01", "2025-11-02", "2025-11-04"][:n],
        "PLAYER_ID": [1] * n,
        "PLAYER_NAME": ["A"] * n,
        "TEAM_ID": [10] * n,
        "TEAM_ABBREVIATION": ["BOS"] * n,
        "MATCHUP": ["BOS vs. NYK", "BOS @ MIA", "BOS vs. LAL"][:n],
        "MIN": [30.0, 28.0, 31.0][:n],
        "PTS": [20, 10, 30][:n],
        "REB": [5, 5, 10][:n],
        "AST": [5, 0, 3][:n],
        "STL": [1, 0, 2][:n],
        "BLK": [0, 1, 0][:n],
        "TOV": [2, 1, 4][:n],
    }
    raw = pd.DataFrame(base)
    raw["FANTASY_PTS"] = [
        fantasy_points(r.PTS, r.REB, r.AST, r.STL, r.BLK, r.TOV) for r in raw.itertuples()
    ]
    return raw


def test_fantasy_formula_matches_the_published_columns():
    assert fantasy_points(20, 5, 5, 1, 0, 2) == pytest.approx(20 + 6 + 7.5 + 3 - 2)


def test_gamelog_parses_opponent_home_and_checks_the_formula():
    out = clean_gamelog(_raw(), "2025-26")
    assert out["opponent"].tolist() == ["NYK", "MIA", "LAL"]
    assert out["home"].tolist() == [True, False, True]
    assert out["game_date"].is_monotonic_increasing
    bad = _raw()
    bad["FANTASY_PTS"] = 0.0
    with pytest.raises(ValueError, match="FANTASY_PTS"):
        clean_gamelog(bad, "2025-26")


def _games(n_games: int = 30, seed: int = 0) -> pd.DataFrame:
    """Two seasons of one team pair repeated, with a few players on each side."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, start in (("2024-25", "2024-10-25"), ("2025-26", "2025-10-25")):
        for g in range(n_games):
            date = pd.Timestamp(start) + pd.Timedelta(days=2 * g)
            for team, opp, home in (("AAA", "BBB", g % 2 == 0), ("BBB", "AAA", g % 2 == 1)):
                for p in range(3):
                    pid = (0 if team == "AAA" else 10) + p
                    rows.append(
                        {
                            "season": season,
                            "game_id": f"{season}-{g}",
                            "game_date": date,
                            "player_id": pid,
                            "team": team,
                            "opponent": opp,
                            "home": home,
                            "minutes": 25 + 3 * p + rng.normal(0, 1),
                            "fantasy": 20 + 8 * p + rng.normal(0, 5),
                        }
                    )
    return pd.DataFrame(rows)


def test_features_use_only_earlier_games_of_the_same_player():
    games = _games()
    base = build_features(games)
    tweaked = games.copy()
    target = tweaked[(tweaked["player_id"] == 1) & (tweaked["season"] == "2025-26")].iloc[10]
    tweaked.loc[
        tweaked["game_id"].eq(target["game_id"]) & tweaked["player_id"].eq(1), "fantasy"
    ] += 50
    other = build_features(tweaked)
    cols = ["season_mean", "last5", "last10_ewm", "min_last5", "n_prior_season"]
    key = lambda d: d[(d["player_id"] == 1) & (d["game_id"] == target["game_id"])][cols]  # noqa: E731
    pd.testing.assert_frame_equal(
        key(base).reset_index(drop=True), key(other).reset_index(drop=True)
    )
    later = base[(base["player_id"] == 1) & (base["season"] == "2025-26")].iloc[11]
    later_other = other[(other["player_id"] == 1) & (other["game_id"] == later["game_id"])].iloc[0]
    assert later_other["last5"] > later["last5"]  # the following game does see the change


def test_season_mean_and_prior_season_mean_are_what_they_say():
    games = _games()
    feats = build_features(games)
    p1 = feats[(feats["player_id"] == 1) & (feats["season"] == "2025-26")].reset_index(drop=True)
    assert np.isnan(p1.loc[0, "season_mean"]) and p1.loc[0, "n_prior_season"] == 0
    assert p1.loc[5, "season_mean"] == pytest.approx(p1.loc[:4, "target"].mean())
    last_season = games[(games["player_id"] == 1) & (games["season"] == "2024-25")][
        "fantasy"
    ].mean()
    assert p1.loc[3, "prior_season_mean"] == pytest.approx(last_season)
    first_season = feats[feats["season"] == "2024-25"]
    assert first_season["prior_season_mean"].isna().all()  # no earlier season in the data


def test_rest_days_back_to_back_and_opponent_strength_use_only_the_past():
    feats = build_features(_games())
    p1 = feats[(feats["player_id"] == 1) & (feats["season"] == "2024-25")].reset_index(drop=True)
    assert p1.loc[0, "rest_days"] == 7 and p1.loc[1, "rest_days"] == 2 and p1.loc[1, "b2b"] == 0
    first_day = feats[(feats["season"] == "2024-25") & (feats["game_id"] == "2024-25-0")]
    assert first_day["opp_allowed_rel"].isna().all()  # nothing earlier to learn from


def test_eligibility_needs_history_minutes_and_fills_missing_values():
    feats = build_features(_games())
    rows = eligible_rows(feats)
    assert (rows["n_prior_total"] >= 5).all() and (rows["min_last5"] >= 15).all()
    filled = fill_missing(rows)
    for col in ("season_mean", "prior_season_mean", "opp_allowed_rel"):
        assert filled[col].notna().all()


def test_next_game_placeholders_use_the_schedule_and_never_leak_their_own_score():
    games = _games(n_games=20).assign(
        team_id=lambda d: np.where(d["team"] == "AAA", 1, 2),
        player=lambda d: d["player_id"].astype(str),
    )
    last = games["game_date"].max()
    schedule = pd.DataFrame(
        {
            "game_id": ["next-1", "old"],
            "game_date": [last + pd.Timedelta(days=2), last - pd.Timedelta(days=5)],
            "played": [False, True],
            "home_team_id": [2, 1],
            "away_team_id": [1, 2],
        }
    )
    with_next = append_next_games(games, schedule, since=last)
    nxt = with_next[with_next["is_next"]]
    assert set(nxt["game_id"]) == {"next-1"} and len(nxt) == 6  # 3 players per team
    a = nxt[nxt["team"] == "AAA"].iloc[0]
    assert a["opponent"] == "BBB" and not a["home"]  # team 1 visits team 2 in the next game
    feats = build_features(with_next)
    row = feats[feats["is_next"] & (feats["player_id"] == 1)].iloc[0]
    played = games[(games["player_id"] == 1) & (games["season"] == "2025-26")]
    assert row["n_prior_season"] == len(played)
    assert row["season_mean"] == pytest.approx(played["fantasy"].mean())
    stale = append_next_games(games, schedule, since=last + pd.Timedelta(days=60))
    assert not stale["is_next"].any()  # nobody played recently: no one is projected


def test_a_traded_player_is_projected_only_for_his_new_team():
    games = _games(n_games=20).assign(
        team_id=lambda d: np.where(d["team"] == "AAA", 1, 2),
        player=lambda d: d["player_id"].astype(str),
    )
    last = games["game_date"].max()
    # player 0 moves from AAA (team 1) to BBB (team 2) for the last game of the window
    moved = (games["player_id"] == 0) & (games["game_date"] == last)
    games.loc[moved, ["team", "team_id", "opponent"]] = ["BBB", 2, "AAA"]
    schedule = pd.DataFrame(
        {
            "game_id": ["next-1", "next-2"],
            "game_date": [last + pd.Timedelta(days=1), last + pd.Timedelta(days=2)],
            "played": [False, False],
            "home_team_id": [1, 2],
            "away_team_id": [3, 4],
        }
    )
    nxt = append_next_games(games, schedule, since=last)
    nxt = nxt[nxt["is_next"]]
    assert not nxt.duplicated("player_id").any()
    traded = nxt[nxt["player_id"] == 0].iloc[0]
    assert traded["team_id"] == 2 and traded["game_id"] == "next-2"
