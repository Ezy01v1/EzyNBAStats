"""Response shapes of the API (read-only, educational data)."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Educational use only. Probabilities come from simple models that do not know about injuries, "
    "rest or lineups; nothing here is betting advice."
)


class Meta(BaseModel):
    season: str
    today: date
    data_as_of: date | None = Field(description="Date of the last finished game in the data")
    refreshed_at: datetime
    stale: bool = Field(description="True if a refresh failed and older data is being served")
    schedule_games_per_team: str
    warnings: list[str]
    elo: dict[str, float]
    disclaimer: str = DISCLAIMER


class Rating(BaseModel):
    team_id: int
    team: str
    conference: str
    elo: float


class Game(BaseModel):
    game_id: str
    date: date
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    played: bool
    home_pts: int | None = None
    away_pts: int | None = None
    p_home: float = Field(description="Elo probability that the home side wins (pre-game)")
    favorite: str
    surprise: bool | None = Field(
        default=None, description="Played games only: the winner had at most 35% pre-game"
    )


class GamesResponse(BaseModel):
    date: date | None
    games: list[Game]
    disclaimer: str = DISCLAIMER


class Standing(BaseModel):
    conference: str
    rank_in_conference: int
    team_id: int
    team: str
    wins: int
    losses: int
    win_pct: float
    last10: str
    point_diff: float | None
    elo: float | None
    p_playoffs: float | None = Field(default=None, description="Only with include_odds=true")


class StandingsResponse(BaseModel):
    season: str
    data_as_of: date | None
    standings: list[Standing]
    disclaimer: str = DISCLAIMER


class SimTeam(BaseModel):
    team_id: int
    team: str
    conference: str
    elo: float
    current_wins: int
    exp_wins: float
    wins_p10: float
    wins_p90: float
    p_playoffs: float
    p_conf_finals: float
    p_finals: float
    p_champion: float


class SimulationResponse(BaseModel):
    season: str
    n_sims: int
    rating_sd: float
    monte_carlo_error: float = Field(description="Largest standard error of any probability")
    data_as_of: date | None
    notes: str
    teams: list[SimTeam]
    disclaimer: str = DISCLAIMER


class Projection(BaseModel):
    player_id: int
    player: str
    team: str
    opponent: str
    home: bool
    game_date: date
    games_this_season: int
    projection: float
    low50: float
    high50: float
    low80: float
    high80: float


class ProjectionsResponse(BaseModel):
    data_as_of: date | None
    note: str = (
        "Conditional on the player playing: the data has no injury information. A single game is "
        "very noisy; use the intervals, not just the point projection."
    )
    projections: list[Projection]
    disclaimer: str = DISCLAIMER
