"""Pre-game features for projecting a player's fantasy points: only games BEFORE the one predicted.

Every rolling quantity is shifted by one game (tests change a game's score and check that its own
features do not move). Rows are "if he plays": games he missed do not exist in the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PRIOR_GAMES = 5  # a row needs this many earlier games (any season) to get a projection
MIN_MINUTES = 15  # average minutes over the last 5 games: below this the role is too unstable
REST_CAP = 7
OPP_PRIOR_GAMES = 20  # shrinkage of the opponent's "fantasy points allowed" towards the league
FEATURES = [
    "season_mean",
    "last5",
    "last10_ewm",
    "prior_season_mean",
    "min_last5",
    "home",
    "rest_days",
    "b2b",
    "opp_allowed_rel",
    "n_prior_season",
]


def _team_allowed(games: pd.DataFrame) -> pd.DataFrame:
    """Fantasy points each team ALLOWED in each game (what the opposing players scored)."""
    per_game = games.groupby(["game_id", "opponent"], as_index=False).agg(
        allowed=("fantasy", "sum"), game_date=("game_date", "first"), season=("season", "first")
    )
    per_game = per_game.rename(columns={"opponent": "team"})
    return per_game.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)


def _league_mean_before_each_date(allowed: pd.DataFrame) -> pd.Series:
    """League-average fantasy points allowed per team-game, using only EARLIER dates of the same
    season (games played the same day are not used: they may be simultaneous)."""
    daily = allowed.groupby(["season", "game_date"]).agg(
        total=("allowed", "sum"), n=("allowed", "size")
    )
    daily = daily.sort_index()
    before = daily.groupby(level="season").cumsum() - daily
    mean_before = before["total"] / before["n"].replace(0, np.nan)
    keys = pd.MultiIndex.from_arrays([allowed["season"], allowed["game_date"]])
    return pd.Series(mean_before.reindex(keys).to_numpy(), index=allowed.index)


def _opponent_strength(games: pd.DataFrame) -> pd.Series:
    """For every player-game: how many more fantasy points than the league average the opponent
    allowed in its earlier games of the same season (shrunk towards 0 with few games)."""
    allowed = _team_allowed(games)
    by = allowed.groupby(["team", "season"])["allowed"]
    prior_sum = by.cumsum() - allowed["allowed"]
    prior_n = by.cumcount()
    league_before = _league_mean_before_each_date(allowed)
    shrunk = (prior_sum + OPP_PRIOR_GAMES * league_before) / (prior_n + OPP_PRIOR_GAMES)
    allowed["opp_allowed_rel"] = shrunk - league_before
    key = allowed.set_index(["game_id", "team"])["opp_allowed_rel"]
    index = pd.MultiIndex.from_arrays([games["game_id"], games["opponent"]])
    return pd.Series(key.reindex(index).to_numpy(), index=games.index)


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add pre-game features (and the ``target`` = fantasy points) to a player-game table.

    Needs: season, game_id, game_date, player_id, team, opponent, home, minutes, fantasy.
    Features use only earlier games of the same player (``season_mean``, ``n_prior_season`` within
    the season; ``last5``, ``last10_ewm``, ``min_last5`` across seasons).
    """
    df = games.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    by_player = df.groupby("player_id")
    by_season = df.groupby(["player_id", "season"])

    df["target"] = df["fantasy"]
    df["n_prior_season"] = by_season.cumcount()
    prior_sum = by_season["fantasy"].cumsum() - df["fantasy"]
    df["season_mean"] = prior_sum / df["n_prior_season"].replace(0, np.nan)

    shifted = by_player["fantasy"].shift(1)
    df["n_prior_total"] = by_player.cumcount()
    df["last5"] = shifted.groupby(df["player_id"]).transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    df["last10_ewm"] = shifted.groupby(df["player_id"]).transform(
        lambda s: s.ewm(span=10, min_periods=3).mean()
    )
    minutes_shifted = by_player["minutes"].shift(1)
    df["min_last5"] = minutes_shifted.groupby(df["player_id"]).transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )

    season_means = (
        df.groupby(["player_id", "season"])
        .agg(mean=("fantasy", "mean"), n=("fantasy", "size"))
        .reset_index()
    )
    season_means = season_means[season_means["n"] >= 10]
    seasons = sorted(df["season"].unique())
    previous = dict(zip(seasons[1:], seasons[:-1], strict=True))
    lookup = season_means.assign(
        season=season_means["season"].map({v: k for k, v in previous.items()})
    )
    df = df.merge(
        lookup[["player_id", "season", "mean"]].rename(columns={"mean": "prior_season_mean"}),
        on=["player_id", "season"],
        how="left",
    )

    days = by_player["game_date"].diff().dt.days
    df["rest_days"] = days.fillna(REST_CAP).clip(upper=REST_CAP)
    df["b2b"] = (df["rest_days"] == 1).astype(int)
    df["home"] = df["home"].astype(int)
    df["opp_allowed_rel"] = _opponent_strength(df).to_numpy()
    return df


def eligible_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Rows we project: enough history, a real role (>= 15 minutes lately), a usable last-5 mean."""
    ok = (
        (features["n_prior_total"] >= MIN_PRIOR_GAMES)
        & (features["min_last5"] >= MIN_MINUTES)
        & features["last5"].notna()
    )
    return features[ok].reset_index(drop=True)


def fill_missing(features: pd.DataFrame) -> pd.DataFrame:
    """No season history yet, or no prior season -> use the 10-game average."""
    out = features.copy()
    out["season_mean"] = out["season_mean"].fillna(out["last10_ewm"])
    out["prior_season_mean"] = out["prior_season_mean"].fillna(out["last10_ewm"])
    out["opp_allowed_rel"] = out["opp_allowed_rel"].fillna(0.0)
    return out


def append_next_games(
    games: pd.DataFrame, schedule: pd.DataFrame, since: pd.Timestamp, active_days: int = 21
) -> pd.DataFrame:
    """Add one placeholder row per active player for his team's next unplayed game.

    ``schedule``: ``clean_schedule`` output (played flag, home/away team ids). A player is active
    if he played in the ``active_days`` before ``since`` (an injured star who has not played
    recently is NOT included: the data has no injury information). The placeholder has
    ``fantasy`` = 0 and ``minutes`` = 0 so that rolling features stay defined; only its features
    are meaningful, and its ``is_next`` flag is True. Features of a placeholder use earlier games
    only, exactly like any other row.
    """
    pending = schedule[~schedule["played"]].sort_values(["game_date", "game_id"])
    abbr = games.drop_duplicates("team_id").set_index("team_id")["team"]
    rows = []
    for team_id, team_games in games[
        games["game_date"] >= since - pd.Timedelta(days=active_days)
    ].groupby("team_id"):
        nxt = pending[(pending["home_team_id"] == team_id) | (pending["away_team_id"] == team_id)]
        if nxt.empty:
            continue
        g = nxt.iloc[0]
        is_home = g["home_team_id"] == team_id
        opp_id = g["away_team_id"] if is_home else g["home_team_id"]
        for player_id, p in team_games.sort_values("game_date").groupby("player_id"):
            last = p.iloc[-1]
            rows.append(
                {
                    "season": last["season"],
                    "game_id": g["game_id"],
                    "game_date": g["game_date"],
                    "player_id": player_id,
                    "player": last["player"],
                    "team_id": team_id,
                    "team": abbr.get(team_id, ""),
                    "opponent": abbr.get(opp_id, str(opp_id)),
                    "home": bool(is_home),
                    "minutes": 0.0,
                    "fantasy": 0.0,
                }
            )
    placeholder = pd.DataFrame(rows)
    out = pd.concat(
        [games.assign(is_next=False), placeholder.assign(is_next=True)], ignore_index=True
    )
    return out.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
