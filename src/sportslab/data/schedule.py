"""Schedule, conferences and playoff results, the inputs a season simulator needs."""

from datetime import timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, LeagueStandingsV3, ScheduleLeagueV2

from sportslab.data.client import NbaClient
from sportslab.data.games import team_logs_to_games

SCHEDULE_COLUMNS = [
    "game_id",
    "game_date",
    "played",
    "home_team_id",
    "away_team_id",
    "home_pts",
    "away_pts",
    "neutral_site",
    "tipoff_utc",
]
FINISHED = 3  # gameStatus: 1 = scheduled, 2 = live, 3 = final
REGULAR_SEASON_PREFIX = "002"  # game ids: 001 preseason, 002 regular season, 004 playoffs...


def clean_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games (played and pending), one row per game, in date order."""
    reg = raw[raw["gameId"].str.startswith(REGULAR_SEASON_PREFIX)]
    out = pd.DataFrame(
        {
            "game_id": reg["gameId"].to_numpy(),
            "game_date": pd.to_datetime(reg["gameDate"], format="%m/%d/%Y %H:%M:%S").to_numpy(),
            "played": (reg["gameStatus"] == FINISHED).to_numpy(),
            "home_team_id": reg["homeTeam_teamId"].astype(int).to_numpy(),
            "away_team_id": reg["awayTeam_teamId"].astype(int).to_numpy(),
            "home_pts": reg["homeTeam_score"].to_numpy(),
            "away_pts": reg["awayTeam_score"].to_numpy(),
            "neutral_site": reg["isNeutral"].astype(bool).to_numpy(),
            # Tip-off time (UTC); missing in old payloads and for games without a set time
            "tipoff_utc": (
                pd.to_datetime(reg["gameDateTimeUTC"], utc=True, errors="coerce").to_numpy()
                if "gameDateTimeUTC" in reg
                else pd.NaT
            ),
        }
    )
    if out["game_id"].duplicated().any():
        raise ValueError("Duplicate game ids in the schedule.")
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)[SCHEDULE_COLUMNS]


def games_per_team(schedule: pd.DataFrame) -> pd.Series:
    """Scheduled regular-season games per team (82 each in a complete schedule)."""
    teams = pd.concat([schedule["home_team_id"], schedule["away_team_id"]])
    return teams.value_counts().sort_index()


def clean_conferences(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.rename(
        columns={
            "TeamID": "team_id",
            "TeamName": "team",
            "Conference": "conference",
            "Division": "division",
        }
    )
    return out[["team_id", "team", "conference", "division"]].reset_index(drop=True)


def load_schedule(client: NbaClient, season: str, max_age: timedelta | None = None) -> pd.DataFrame:
    (raw, _) = client.fetch(ScheduleLeagueV2, season=season, max_age=max_age)
    return clean_schedule(raw)


def load_conferences(client: NbaClient, season: str) -> pd.DataFrame:
    (raw,) = client.fetch(LeagueStandingsV3, season=season, season_type="Regular Season")
    return clean_conferences(raw)


def load_playoff_games(client: NbaClient, season: str) -> pd.DataFrame:
    """Playoff games (not play-in) as a one-row-per-game table, like ``load_games``."""
    (log,) = client.fetch(LeagueGameLog, season=season, season_type_all_star="Playoffs")
    return team_logs_to_games(log, season)


def playoff_outcomes(playoff_games: pd.DataFrame) -> dict:
    """Who made the playoffs, who reached the finals and who won, from playoff games alone.

    Teams that lost the play-in are not in playoff games, so ``made_playoffs`` is exactly the
    set of 16 teams that played a series. The finals are the series containing the last game.
    """
    if playoff_games.empty:
        return {"made_playoffs": set(), "finalists": set(), "champion": None}
    g = playoff_games.sort_values(["game_date", "game_id"])
    pair = g.apply(lambda r: frozenset((r["home_team_id"], r["away_team_id"])), axis=1)
    last_pair = pair.iloc[-1]
    finals = g[pair == last_pair]
    wins = pd.concat(
        [
            finals.loc[finals["home_win"] == 1, "home_team_id"],
            finals.loc[finals["home_win"] == 0, "away_team_id"],
        ]
    ).value_counts()
    return {
        "made_playoffs": set(g["home_team_id"]) | set(g["away_team_id"]),
        "finalists": set(last_pair),
        "champion": int(wins.idxmax()),
    }


def schedule_to_games(schedule: pd.DataFrame, season: str) -> pd.DataFrame:
    """Played rows of a schedule as a game table (same shape as ``load_games``).

    Using the schedule for both "already played" and "still to play" keeps the two consistent.
    """
    played = schedule[schedule["played"]]
    out = pd.DataFrame(
        {
            "game_id": played["game_id"],
            "season": season,
            "game_date": played["game_date"],
            "home_team_id": played["home_team_id"],
            "away_team_id": played["away_team_id"],
            "home_pts": played["home_pts"],
            "away_pts": played["away_pts"],
            "home_win": (played["home_pts"] > played["away_pts"]).astype(int),
            "neutral_site": played["neutral_site"],
        }
    )
    return out.reset_index(drop=True)


def games_owed(schedule: pd.DataFrame, teams: pd.Series, per_team: int = 82) -> dict[int, int]:
    """Games each team still has to play that the schedule does not list (``per_team`` minus the
    scheduled games, never negative). Unknown opponents (id 0) do not count as a team."""
    scheduled = games_per_team(schedule).reindex(teams).fillna(0).astype(int)
    return {int(t): int(max(per_team - n, 0)) for t, n in scheduled.items()}
