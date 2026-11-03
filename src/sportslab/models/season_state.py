"""Everything the simulator needs about a season in progress, built from one schedule table."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sportslab.data.schedule import games_owed, games_per_team, schedule_to_games
from sportslab.models.elo import EloParams, final_ratings


@dataclass
class SeasonState:
    ratings: dict[int, float]
    conferences: pd.DataFrame
    current_wins: dict[int, int]
    remaining: pd.DataFrame
    extra_games: dict[int, int]
    games_played: int
    as_of: pd.Timestamp | None
    scheduled_min: int = 82
    scheduled_max: int = 82
    owed_total: int = 0
    warnings: list[str] = field(default_factory=list)


def build_season_state(
    history: pd.DataFrame,
    schedule: pd.DataFrame,
    conferences: pd.DataFrame,
    season: str,
    elo_params: EloParams,
) -> SeasonState:
    """Ratings after the games already played, standings, the games left and the games the
    schedule does not list (a warning is raised for those instead of hiding them).

    ``history``: finished games of earlier seasons. ``schedule``: ``clean_schedule`` output for
    ``season``, both played and pending games, so the two parts always agree.
    """
    played = schedule_to_games(schedule, season)
    ratings = final_ratings(pd.concat([history, played], ignore_index=True), elo_params)
    wins = pd.concat(
        [
            played.loc[played["home_win"] == 1, "home_team_id"],
            played.loc[played["home_win"] == 0, "away_team_id"],
        ]
    ).value_counts()
    teams = conferences["team_id"]
    remaining = schedule.loc[~schedule["played"], ["home_team_id", "away_team_id", "neutral_site"]]
    owed = games_owed(schedule, teams)

    warnings = []
    scheduled = games_per_team(schedule).reindex(teams).fillna(0).astype(int)
    if (scheduled < 82).any():
        warnings.append(
            f"The schedule lists {scheduled.min()}-{scheduled.max()} games per team instead of 82: "
            f"the other {sum(owed.values())} team-games are played against an average opponent."
        )
    if int(wins.sum()) != len(played):
        warnings.append("Played games and recorded wins disagree.")
    return SeasonState(
        ratings=ratings,
        conferences=conferences,
        current_wins={int(t): int(wins.get(t, 0)) for t in teams},
        remaining=remaining.reset_index(drop=True),
        extra_games=owed,
        games_played=len(played),
        as_of=played["game_date"].max() if len(played) else None,
        scheduled_min=int(scheduled.min()),
        scheduled_max=int(scheduled.max()),
        owed_total=int(sum(owed.values())),
        warnings=warnings,
    )
