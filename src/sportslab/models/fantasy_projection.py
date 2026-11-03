"""Next-game fantasy projections for every active player, shared by the Streamlit page and the API.

Fit on finished seasons, calibrate the intervals on the most recent finished one (out of sample),
then project each active player's team's next scheduled game. Rows are "if he plays": the data
has no injury information.
"""

from __future__ import annotations

import pandas as pd

from sportslab.features.fantasy import (
    append_next_games,
    build_features,
    eligible_rows,
    fill_missing,
)
from sportslab.models.fantasy_model import IntervalCalibrator, fit_ridge, predict_ridge

PROJECTION_COLUMNS = [
    "player_id",
    "player",
    "team",
    "opponent",
    "home",
    "game_date",
    "n_prior_season",
    "season_mean",
    "projection",
    "low50",
    "high50",
    "low80",
    "high80",
]


def project_next_games(
    games: pd.DataFrame,
    schedule: pd.DataFrame,
    train_seasons: list[str],
    calibration_season: str,
) -> pd.DataFrame:
    """One row per active player with a projection and 50% / 80% intervals for his next game.

    ``games``: player games (``load_player_games``) including the season in progress.
    ``schedule``: ``clean_schedule`` output of that season.
    """
    since = games["game_date"].max()
    feats = build_features(append_next_games(games, schedule, since))
    played = feats[~feats["is_next"]]
    train = fill_missing(eligible_rows(played[played["season"].isin(train_seasons)]))
    ridge = fit_ridge(train)
    calib_rows = fill_missing(eligible_rows(played[played["season"] == calibration_season]))
    calibrator = IntervalCalibrator().fit(predict_ridge(ridge, calib_rows), calib_rows["target"])
    nxt = fill_missing(eligible_rows(feats[feats["is_next"]]))
    out = nxt[
        [
            "player_id",
            "player",
            "team",
            "opponent",
            "home",
            "game_date",
            "n_prior_season",
            "season_mean",
        ]
    ].copy()
    out["home"] = out["home"].astype(bool)
    out["projection"] = predict_ridge(ridge, nxt)
    for level in (0.5, 0.8):
        low, high = calibrator.interval(out["projection"].to_numpy(), level)
        out[f"low{int(level * 100)}"], out[f"high{int(level * 100)}"] = low, high
    return out[PROJECTION_COLUMNS].reset_index(drop=True)
