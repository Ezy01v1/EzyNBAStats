"""The Module 2 logistic regression (Elo + form + rest), used live on games not played yet.

Training rows come from ``build_pregame_features`` (finished games only). For a pending game the
same features are rebuilt without its result: rest counts days since the team's previous
*scheduled* game, form uses only finished games of the season, and the Elo difference comes from
the current ratings. Neutral-site games are left out, as in Module 2 (their "home" is arbitrary).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportslab.features.team_form import _add_rest_and_form, _team_game_table
from sportslab.models.supervised import ALL_FEATURES, make_logistic

LOGISTIC_MODEL = "logistic-form-rest-v1"


def fit_logistic(features: pd.DataFrame):
    """Fit on finished, non-neutral games (``build_pregame_features`` output)."""
    rows = features[~features["neutral_site"]].dropna(subset=ALL_FEATURES)
    return make_logistic().fit(rows[ALL_FEATURES], rows["home_win"])


def pending_features(
    schedule: pd.DataFrame, season: str, ratings: dict[int, float], initial: float = 1500.0
) -> pd.DataFrame:
    """Features of every unplayed game of ``schedule`` (``clean_schedule`` output)."""
    games = schedule.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    games = games.assign(season=season)
    unplayed = ~games["played"]
    # No result yet: a missing margin is skipped by the form window, rest still counts the date
    games.loc[unplayed, ["home_pts", "away_pts"]] = np.nan
    long = _add_rest_and_form(_team_game_table(games))
    home = long[long["side"] == "home"].set_index("row")
    away = long[long["side"] == "away"].set_index("row")
    out = games.copy()
    for side, part in (("home", home), ("away", away)):
        out[f"{side}_rest"] = part["rest"].reindex(out.index).to_numpy()
        out[f"{side}_form"] = part["form"].reindex(out.index).to_numpy()
        out[f"{side}_b2b"] = (out[f"{side}_rest"] == 1).astype(int)
    r_home = out["home_team_id"].map(lambda t: ratings.get(t, initial))
    r_away = out["away_team_id"].map(lambda t: ratings.get(t, initial))
    out["elo_diff"] = r_home - r_away
    out["form_diff"] = out["home_form"] - out["away_form"]
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    return out[unplayed].reset_index(drop=True)


def predict_pending(model, pending: pd.DataFrame) -> pd.DataFrame:
    """``pending`` (non-neutral games only) with the model's ``p_home``."""
    out = pending[~pending["neutral_site"]].reset_index(drop=True)
    out["p_home"] = model.predict_proba(out[ALL_FEATURES])[:, 1] if len(out) else []
    return out
