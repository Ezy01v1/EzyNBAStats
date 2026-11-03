"""Player awards from NBA.com (PlayerAwards, one request per player, full career history).

Labels we use: All-Star selection, All-NBA team (1st/2nd/3rd) and MVP. NBA.com gives only MVP
winners (no vote shares), so MVP is a ranking question, not a modeling one.
"""

from collections.abc import Iterable

import pandas as pd
from nba_api.stats.endpoints import PlayerAwards

from sportslab.data.client import NbaClient

AWARD_COLUMNS = ["player_id", "season", "award", "all_nba_team"]
ALL_STAR = "NBA All-Star"
ALL_NBA = "All-NBA"
MVP = "NBA Most Valuable Player"


def clean_awards(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per award: player_id, season (e.g. 2018-19), award name, All-NBA team number."""
    if raw.empty:
        return pd.DataFrame(columns=AWARD_COLUMNS)
    out = raw.rename(
        columns={
            "PERSON_ID": "player_id",
            "SEASON": "season",
            "DESCRIPTION": "award",
            "ALL_NBA_TEAM_NUMBER": "all_nba_team",
        }
    )
    out["all_nba_team"] = pd.to_numeric(out["all_nba_team"], errors="coerce")
    return out[AWARD_COLUMNS].reset_index(drop=True)


def load_player_awards(client: NbaClient, player_id: int) -> pd.DataFrame:
    (raw,) = client.fetch(PlayerAwards, player_id=int(player_id))
    return clean_awards(raw)


def load_awards_for(client: NbaClient, player_ids: Iterable[int]) -> pd.DataFrame:
    """Awards of many players (one cached request each; ~1 s the first time)."""
    frames = [load_player_awards(client, pid) for pid in player_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=AWARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def award_labels(awards: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season) flags: ``all_star``, ``all_nba`` (any team), ``all_nba_first``,
    ``mvp``."""
    if awards.empty:
        return pd.DataFrame(
            columns=["player_id", "season", "all_star", "all_nba", "all_nba_first", "mvp"]
        )
    a = awards.assign(
        all_star=awards["award"] == ALL_STAR,
        all_nba=awards["award"] == ALL_NBA,
        all_nba_first=(awards["award"] == ALL_NBA) & (awards["all_nba_team"] == 1),
        mvp=awards["award"] == MVP,
    )
    flags = ["all_star", "all_nba", "all_nba_first", "mvp"]
    return a.groupby(["player_id", "season"], as_index=False)[flags].any()


def prior_selections(awards: pd.DataFrame, award: str = ALL_STAR) -> pd.DataFrame:
    """For every (player, season) seen in ``awards`` for ``award``: the number of EARLIER seasons
    the player got it. Built from the whole career, so it does not depend on our data window."""
    sel = awards.loc[awards["award"] == award, ["player_id", "season"]].drop_duplicates()
    sel = sel.sort_values(["player_id", "season"]).reset_index(drop=True)
    sel["prior_count"] = sel.groupby("player_id").cumcount()
    return sel


def selections_before(awards: pd.DataFrame, season: str, award: str = ALL_STAR) -> pd.Series:
    """player_id -> number of seasons strictly before ``season`` in which he got ``award``."""
    sel = awards.loc[(awards["award"] == award) & (awards["season"] < season)]
    return sel.drop_duplicates(["player_id", "season"]).groupby("player_id").size()
