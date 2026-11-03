"""Data the API serves, loaded through the single NbaClient and kept in memory with a TTL.

Requests never talk to NBA.com directly: they read from here. A refresh goes through the client
(rate limit, retries, disk cache). If a refresh fails but older data exists, the old data is
served and marked ``stale``; with no data at all the service raises ``ServiceUnavailable``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from sportslab.data import NbaClient, NbaClientError
from sportslab.data.gamelog import load_player_games
from sportslab.data.games import load_games, season_labels
from sportslab.data.schedule import load_conferences, load_schedule, schedule_to_games
from sportslab.models.elo import EloParams, run_elo
from sportslab.models.fantasy_projection import project_next_games
from sportslab.models.season_sim import VALIDATED_RATING_SD, SimConfig, simulate_season
from sportslab.models.season_state import SeasonState, build_season_state

CURRENT = "2026-27"
TUNED_ELO = EloParams(k=24, home_advantage=40, carryover=0.7)
FINISHED_PLAYER_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]
FRESHNESS = timedelta(hours=3)  # how old NBA.com data may be before the client re-requests it
MAX_SIMS = 20_000


class ServiceUnavailable(RuntimeError):
    """No data could be loaded (and there is no older copy to fall back on)."""


@dataclass
class CurrentContext:
    """Everything built from the schedule of the season in progress."""

    season: str
    schedule: pd.DataFrame
    conferences: pd.DataFrame
    state: SeasonState
    pre_game: pd.DataFrame  # run_elo output: pre-game p_home for every game played so far
    loaded_at: datetime = field(default_factory=datetime.now)


def load_context(client: NbaClient, season: str = CURRENT) -> CurrentContext:
    history = load_games(client, season_labels(2015, int(season[:4]) - 1))
    schedule = load_schedule(client, season, max_age=FRESHNESS)
    conferences = load_conferences(client, season)
    state = build_season_state(history, schedule, conferences, season, TUNED_ELO)
    played = schedule_to_games(schedule, season)
    pre = run_elo(pd.concat([history, played], ignore_index=True), TUNED_ELO)
    return CurrentContext(season, schedule, conferences, state, pre)


class DataService:
    """In-memory, thread-safe holder of the context, simulations and projections."""

    def __init__(
        self,
        client: NbaClient | None = None,
        context_loader: Callable[[NbaClient], CurrentContext] | None = None,
        ttl: timedelta = timedelta(minutes=30),
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client = client or NbaClient()
        self._loader = context_loader or load_context
        self._ttl = ttl.total_seconds()
        self._clock = clock
        self._lock = threading.Lock()
        self._context: CurrentContext | None = None
        self._loaded_at = 0.0
        self._stale = False
        self._simulations: dict[tuple, pd.DataFrame] = {}
        self._projections: tuple[str, pd.DataFrame] | None = None

    def context(self) -> tuple[CurrentContext, bool]:
        """The context and whether it is stale (a refresh failed and an old copy is served)."""
        with self._lock:
            expired = self._context is None or self._clock() - self._loaded_at > self._ttl
            if expired:
                try:
                    self._context = self._loader(self._client)
                    self._loaded_at = self._clock()
                    self._stale = False
                except NbaClientError as exc:
                    if self._context is None:
                        raise ServiceUnavailable(
                            "Could not load data from NBA.com and there is no cached copy. "
                            "stats.nba.com blocks many cloud IPs; run this service locally."
                        ) from exc
                    self._stale = True
            return self._context, self._stale

    def simulation(self, n_sims: int, rating_sd: float = VALIDATED_RATING_SD) -> pd.DataFrame:
        """Season simulation table (team_id index), cached per data version and settings."""
        ctx, _ = self.context()
        key = (str(ctx.loaded_at), n_sims, rating_sd)
        with self._lock:
            if key not in self._simulations:
                self._simulations.clear()  # keep only the current data version
                state = ctx.state
                table = simulate_season(
                    state.ratings,
                    state.conferences,
                    state.current_wins,
                    state.remaining,
                    state.extra_games,
                    SimConfig(
                        home_advantage=TUNED_ELO.home_advantage,
                        n_sims=n_sims,
                        rating_sd=rating_sd,
                        seed=0,
                    ),
                ).table
                self._simulations[key] = table
            return self._simulations[key]

    def projections(self) -> pd.DataFrame:
        """Next-game fantasy projections for active players (cached per data version)."""
        ctx, _ = self.context()
        version = str(ctx.loaded_at)
        with self._lock:
            if self._projections is None or self._projections[0] != version:
                try:
                    finished = load_player_games(self._client, FINISHED_PLAYER_SEASONS)
                    current = load_player_games(self._client, [ctx.season], max_age=FRESHNESS)
                except NbaClientError as exc:
                    raise ServiceUnavailable("Could not load the player game logs.") from exc
                games = pd.concat([finished, current], ignore_index=True)
                table = project_next_games(
                    games, ctx.schedule, TRAIN_SEASONS, FINISHED_PLAYER_SEASONS[-1]
                )
                self._projections = (version, table)
            return self._projections[1]
