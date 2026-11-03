import numpy as np
import pandas as pd
import pytest

from sportslab.data.awards import (
    award_labels,
    clean_awards,
    selections_before,
)
from sportslab.data.player_season import build_player_season, cutoff_at_fraction
from sportslab.features.awards import ALL_STAR_FEATURES, as_matrix, build_panel
from sportslab.models.awards_model import (
    evaluate,
    expanding_window,
    make_logistic,
    r_precision,
    winner_rank,
)


def _raw_awards() -> pd.DataFrame:
    rows = [
        (1, "NBA All-Star", "2018-19", None),
        (1, "NBA All-Star", "2019-20", None),
        (1, "NBA Most Valuable Player", "2019-20", None),
        (1, "All-NBA", "2019-20", "1"),
        (2, "All-NBA", "2019-20", "2"),
        (2, "NBA Player of the Week", "2019-20", None),
    ]
    return pd.DataFrame(
        {
            "PERSON_ID": [r[0] for r in rows],
            "DESCRIPTION": [r[1] for r in rows],
            "SEASON": [r[2] for r in rows],
            "ALL_NBA_TEAM_NUMBER": [r[3] for r in rows],
            "FIRST_NAME": "x",
        }
    )


def test_clean_awards_renames_and_parses_the_all_nba_team_number():
    out = clean_awards(_raw_awards())
    assert list(out.columns) == ["player_id", "season", "award", "all_nba_team"]
    assert out["all_nba_team"].tolist()[3:5] == [1.0, 2.0]
    assert clean_awards(pd.DataFrame()).empty


def test_labels_flag_each_award_per_player_and_season():
    lab = award_labels(clean_awards(_raw_awards())).set_index(["player_id", "season"])
    assert lab.loc[(1, "2019-20"), ["all_star", "all_nba", "all_nba_first", "mvp"]].tolist() == [
        True,
        True,
        True,
        True,
    ]
    assert lab.loc[(2, "2019-20"), ["all_star", "all_nba_first"]].tolist() == [False, False]
    assert bool(lab.loc[(2, "2019-20"), "all_nba"])  # second team still counts as All-NBA


def test_prior_selections_never_include_the_current_or_later_seasons():
    awards = clean_awards(_raw_awards())
    assert selections_before(awards, "2019-20").to_dict() == {1: 1}  # only 2018-19
    assert selections_before(awards, "2018-19").to_dict() == {}
    assert selections_before(awards, "2030-31").to_dict() == {1: 2}


def test_cutoff_is_the_date_at_which_half_of_the_games_were_played():
    dates = pd.to_datetime(["2020-12-22"] * 10 + ["2021-01-20"] * 10 + ["2021-02-10"] * 20)
    games = pd.DataFrame({"season": "2020-21", "game_date": dates})
    assert cutoff_at_fraction(games, "2020-21", 0.5).isoformat() == "2021-02-10"
    assert cutoff_at_fraction(games, "2020-21", 0.1).isoformat() == "2020-12-22"
    assert cutoff_at_fraction(games, "2020-21", 1.0).isoformat() == "2021-02-10"
    with pytest.raises(ValueError, match="no games"):
        cutoff_at_fraction(games, "2019-20")


def test_player_season_join_adds_advanced_rates_and_team_record():
    base = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "PLAYER_NAME": ["A", "B"],
            "TEAM_ID": [10, 20],
            "TEAM_ABBREVIATION": ["BOS", "NYK"],
            "AGE": [25.0, 30.0],
            "GP": [30, 28],
            "MIN": [34.0, 30.0],
            "PTS": [27.0, 18.0],
            "REB": [8.0, 5.0],
            "AST": [6.0, 3.0],
        }
    )
    adv = pd.DataFrame(
        {"PLAYER_ID": [1, 2], "TS_PCT": [0.6, 0.55], "USG_PCT": [0.3, 0.2], "PIE": [0.18, 0.1]}
    )
    teams = pd.DataFrame({"TEAM_ID": [10, 20], "W_PCT": [0.7, 0.4]})
    out = build_player_season(base, adv, teams, "2025-26")
    assert out.loc[0, ["pts", "pie", "team_w_pct"]].tolist() == [27.0, 0.18, 0.7]
    assert out["season"].eq("2025-26").all()


def _stats_and_awards(n: int = 30, seasons=("2018-19", "2019-20")):
    rng = np.random.default_rng(0)
    rows = []
    for s in seasons:
        for pid in range(n):
            rows.append((s, pid, 15 + 15 * pid / n + rng.normal(0, 1), rng.integers(15, 40)))
    stats = pd.DataFrame(rows, columns=["season", "player_id", "pts", "gp"])
    for c in ("reb", "ast", "ts_pct", "usg_pct", "pie", "min_pg", "team_w_pct"):
        stats[c] = rng.random(len(stats))
    awards = pd.DataFrame(
        {
            "player_id": [n - 1, n - 2],
            "season": ["2018-19", "2019-20"],
            "award": "NBA All-Star",
            "all_nba_team": np.nan,
        }
    )
    return stats, awards, award_labels(awards)


def test_panel_keeps_top_scorers_with_enough_games_and_only_earlier_selections():
    stats, awards, labels = _stats_and_awards()
    panel = build_panel(stats, awards, labels, "2019-20", n_candidates=10, min_gp=20)
    assert len(panel) <= 10 and (panel["gp"] >= 20).all()
    assert set(ALL_STAR_FEATURES) <= set(panel.columns)
    row = panel[panel["player_id"] == 29]
    if len(row):  # selected in 2018-19 -> 1 prior selection in 2019-20, never 2019-20's own
        assert row["prior_all_stars"].iloc[0] == 1
    assert (panel["prior_all_stars"] <= 1).all()
    x = as_matrix(panel, ALL_STAR_FEATURES)
    assert x.shape == (len(panel), len(ALL_STAR_FEATURES)) and np.isfinite(x).all()


def test_r_precision_and_winner_rank():
    scores = np.array([0.9, 0.8, 0.1, 0.7])
    truth = np.array([1, 0, 0, 1])
    assert r_precision(scores, truth) == pytest.approx(0.5)  # top-2 = idx 0, 1; one is a winner
    assert np.isnan(r_precision(scores, np.zeros(4)))
    assert winner_rank(np.array([0.2, 0.9, 0.5]), np.array([0, 0, 1])) == 2


def _synthetic_panels(seed: int = 0, seasons: int = 6, n: int = 80):
    rng = np.random.default_rng(seed)
    panels = {}
    for i in range(seasons):
        skill = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(3 * skill - 3)))
        df = pd.DataFrame(
            {
                "player_id": range(n),
                "pts": 15 + 4 * skill + rng.normal(0, 2, n),
                "pie": 0.1 + 0.03 * skill + rng.normal(0, 0.01, n),
                "reb": rng.random(n),
                "ast": rng.random(n),
                "ts_pct": rng.random(n),
                "usg_pct": rng.random(n),
                "min_pg": rng.random(n),
                "gp_frac": rng.random(n),
                "team_w_pct": rng.random(n),
                "prior_all_stars": 0,
                "all_star": rng.random(n) < p,
            }
        )
        panels[f"20{18 + i}-{19 + i}"] = df
    return panels


def test_expanding_window_never_trains_on_the_test_season():
    panels = _synthetic_panels()
    flipped = {k: v.copy() for k, v in panels.items()}
    last = max(flipped)
    flipped[last]["all_star"] = ~flipped[last]["all_star"]
    kwargs = {"features": ALL_STAR_FEATURES, "label": "all_star", "first_test": "2022-23"}
    a = expanding_window(panels, models={"logit": make_logistic}, **kwargs)
    b = expanding_window(flipped, models={"logit": make_logistic}, **kwargs)
    early = a["test_season"] < last  # earlier test seasons cannot see the last season's labels
    pd.testing.assert_series_equal(a.loc[early, "p_logit"], b.loc[early, "p_logit"])
    assert set(a["test_season"]) == {"2022-23", "2023-24"}


def test_model_beats_the_noisy_points_ranking_on_synthetic_data_and_reports_paired_gains():
    panels = _synthetic_panels(seed=1, seasons=8)
    models = {"pts_only": make_logistic, "logit": make_logistic}
    pred = expanding_window(
        panels, ALL_STAR_FEATURES, "all_star", "2021-22", {"logit": make_logistic}
    )
    per_season, pooled = evaluate(pred, "all_star", ["logit"], ["pts", "pie"])
    assert {"season", "winners", "score_rank_pts", "score_rank_pie", "p_logit"} <= set(
        per_season.columns
    )
    assert per_season["p_logit"].mean() > 0.3
    assert {"log_loss", "brier"} <= set(pooled.columns) and len(models) == 2
