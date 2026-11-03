import numpy as np
import pandas as pd
import pytest

from sportslab.models.elo import EloParams
from sportslab.models.season_backtest import backtest_season, summarize_backtest


def _season_games(season: str, strength: np.ndarray, seed: int, rounds: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = list(range(1, 31))
    rows = []
    for _ in range(rounds):
        for a in ids:
            for b in ids:
                if a < b:
                    p = 1 / (1 + 10 ** (-(strength[a - 1] - strength[b - 1] + 40) / 400))
                    rows.append((season, a, b, int(rng.random() < p)))
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=["season", "home_team_id", "away_team_id", "home_win"])
    df["game_id"] = [f"{season}-{i:05d}" for i in range(len(df))]
    df["game_date"] = pd.Timestamp("2025-10-20") + pd.to_timedelta(np.arange(len(df)) // 6, "D")
    df["neutral_site"] = False
    return df


def _setup():
    strength = np.linspace(-250, 250, 30)
    games = pd.concat(
        [_season_games("2024-25", strength, 1), _season_games("2025-26", strength, 2)],
        ignore_index=True,
    )
    conf = pd.DataFrame({"team_id": range(1, 31), "conference": ["East"] * 15 + ["West"] * 15})
    outcome = {"made_playoffs": set(range(15, 31)), "finalists": {29, 30}, "champion": 30}
    return games, conf, outcome


def test_backtest_uses_only_the_games_before_the_cutoff_and_reports_the_truth():
    games, conf, outcome = _setup()
    frame = backtest_season(
        games, "2025-26", [0.25, 0.5], conf, outcome, EloParams(k=24, home_advantage=40), n_sims=300
    )
    assert set(frame["cutoff"]) == {0.25, 0.5} and len(frame) == 60
    season_games = games[games["season"] == "2025-26"]
    for f in (0.25, 0.5):
        part = frame[frame["cutoff"] == f]
        assert part["current_wins"].sum() == round(f * len(season_games))
        assert part["actual_wins"].sum() == len(season_games)
        assert (part["games_played"] == round(f * len(season_games))).all()
    assert frame.loc[frame["is_champion"], "team_id"].eq(30).all()
    assert frame["made_playoffs"].sum() == 16 * 2


def test_backtest_rejects_nothing_but_never_peeks_at_the_future():
    games, conf, outcome = _setup()
    flipped = games.copy()
    late = (flipped["season"] == "2025-26") & (flipped.index >= len(games) * 3 // 4)
    flipped.loc[late, "home_win"] = 1 - flipped.loc[late, "home_win"]
    args = (conf, outcome, EloParams(k=24, home_advantage=40))
    a = backtest_season(games, "2025-26", [0.25], *args, n_sims=300)
    b = backtest_season(flipped, "2025-26", [0.25], *args, n_sims=300)
    # Elo and current wins at a 25% cutoff cannot depend on the last quarter of the season
    pd.testing.assert_series_equal(a["elo"], b["elo"])
    pd.testing.assert_series_equal(a["current_wins"], b["current_wins"])


def test_summary_computes_coverage_brier_and_title_score_from_known_values():
    frame = pd.DataFrame(
        {
            "season": "2025-26",
            "cutoff": 0.5,
            "team_id": [1, 2, 3, 4],
            "actual_wins": [50, 40, 30, 20],
            "wins_p25": [48, 41, 25, 18],
            "wins_p75": [52, 45, 35, 22],
            "wins_p10": [46, 38, 20, 15],
            "wins_p90": [54, 46, 36, 25],
            "wins_p05": [44, 36, 18, 12],
            "wins_p95": [56, 48, 38, 26],
            "exp_wins": [50, 42, 30, 20],
            "naive_wins": [45, 40, 35, 20],
            "p_playoffs": [1.0, 0.5, 0.5, 0.0],
            "made_playoffs": [True, True, False, False],
            "top8": [True, True, True, False],
            "p_champion": [0.5, 0.3, 0.2, 0.0],
            "is_champion": [False, True, False, False],
        }
    )
    row = summarize_backtest(frame, n_sims=1000).iloc[0]
    assert row["coverage_50%"] == pytest.approx(0.75)  # all but team 2 (40 is outside 41-45)
    assert row["coverage_80%"] == pytest.approx(1.0)
    assert row["mae_wins_sim"] == pytest.approx(0.5) and row["mae_wins_naive"] == pytest.approx(2.5)
    assert row["brier_playoffs_sim"] == pytest.approx((0 + 0.25 + 0.25 + 0) / 4)
    assert row["logscore_title"] == pytest.approx(-np.log(0.3))
    assert row["champion_rank"] == 2 and row["top_title_prob"] == pytest.approx(0.5)
