import numpy as np
import pandas as pd

from sportslab.models.experiment import run_elo_experiment


def _synthetic_games(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 100, size=8)
    rows = []
    for s, season in enumerate(["2021-22", "2022-23", "2023-24", "2024-25"]):
        for d in range(120):
            home, away = rng.choice(8, size=2, replace=False)
            p = 1 / (1 + 10 ** (-(strength[home] + 40 - strength[away]) / 400))
            rows.append(
                {
                    "season": season,
                    "game_date": pd.Timestamp("2021-10-20") + pd.Timedelta(days=365 * s + d),
                    "home_team_id": int(home),
                    "away_team_id": int(away),
                    "home_win": int(rng.random() < p),
                    "neutral_site": False,
                }
            )
    return pd.DataFrame(rows)


def test_experiment_scores_every_model_on_the_test_season_only():
    exp = run_elo_experiment(
        _synthetic_games(),
        validation=["2023-24"],
        test="2024-25",
        ks=[10, 30],
        home_advantages=[0, 60],
        carryovers=[0.7],
    )
    assert set(exp.scores.index) == {"coin 50%", "home rate (train)", "elo default", "elo tuned"}
    assert {"log_loss", "brier", "accuracy"} <= set(exp.scores.columns)
    assert set(exp.test_predictions["season"]) == {"2024-25"}
    assert exp.tuned.k in (10, 30)


def test_experiment_tuning_never_sees_the_test_season():
    games = _synthetic_games()
    flipped = games.copy()
    test = flipped["season"] == "2024-25"
    flipped.loc[test, "home_win"] = (
        1 - flipped.loc[test, "home_win"]
    )  # corrupt only the test labels
    kwargs = {
        "validation": ["2023-24"],
        "test": "2024-25",
        "ks": [10, 30],
        "home_advantages": [0, 60],
        "carryovers": [0.7],
    }
    a = run_elo_experiment(games, **kwargs)
    b = run_elo_experiment(flipped, **kwargs)
    assert a.tuned == b.tuned
    pd.testing.assert_frame_equal(a.grid, b.grid)
