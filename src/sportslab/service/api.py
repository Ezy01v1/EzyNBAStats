"""Read-only HTTP API over the Sports Analytics Lab models (FastAPI).

Run locally::

    python -m uv run uvicorn sportslab.service.api:app --host 127.0.0.1 --port 8000

By design it connects to nothing else: it reads NBA.com through the shared client (rate limit,
retries, disk cache), computes in memory and answers JSON. There is no authentication, CORS is not
enabled and it binds to localhost; do not expose it publicly as it is.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from sportslab.analysis.daily import (
    add_probabilities,
    games_on,
    last_game_day,
    next_game_day,
    standings,
    surprises,
)
from sportslab.data.schedule import schedule_to_games
from sportslab.service import schemas
from sportslab.service.context import (
    MAX_SIMS,
    TUNED_ELO,
    DataService,
    ServiceUnavailable,
)

_service: DataService | None = None


def get_service() -> DataService:
    """The shared service (created on first use so importing the app loads nothing)."""
    global _service
    if _service is None:
        _service = DataService()
    return _service


def _none_if_nan(value):
    return None if pd.isna(value) else value


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sports Analytics Lab API",
        version="0.1.0",
        description=(
            "Read-only, educational NBA analytics: Elo game probabilities, standings, season "
            "simulation and fantasy projections with intervals. Not betting advice."
        ),
    )

    @app.exception_handler(ServiceUnavailable)
    def unavailable(_request, exc: ServiceUnavailable):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness only: does not load any data."""
        return {"status": "ok"}

    @app.get("/v1/meta", response_model=schemas.Meta)
    def meta(service: DataService = Depends(get_service)):
        ctx, stale = service.context()
        state = ctx.state
        return schemas.Meta(
            season=ctx.season,
            today=date.today(),
            data_as_of=_to_date(last_game_day(ctx.schedule)),
            refreshed_at=ctx.loaded_at,
            stale=stale,
            schedule_games_per_team=f"{state.scheduled_min}-{state.scheduled_max}",
            warnings=state.warnings,
            elo={
                "k": TUNED_ELO.k,
                "home_advantage": TUNED_ELO.home_advantage,
                "carryover": TUNED_ELO.carryover,
            },
        )

    @app.get("/v1/ratings", response_model=list[schemas.Rating])
    def ratings(service: DataService = Depends(get_service)):
        ctx, _ = service.context()
        conf = ctx.conferences
        rows = [
            schemas.Rating(
                team_id=int(r.team_id),
                team=r.team,
                conference=r.conference,
                elo=float(ctx.state.ratings.get(int(r.team_id), 1500.0)),
            )
            for r in conf.itertuples()
        ]
        return sorted(rows, key=lambda r: -r.elo)

    @app.get("/v1/games", response_model=schemas.GamesResponse)
    def games(
        day: date | None = Query(
            default=None,
            alias="date",
            description="YYYY-MM-DD; default = the next day with games (or the last one)",
        ),
        service: DataService = Depends(get_service),
    ):
        ctx, _ = service.context()
        schedule = ctx.schedule
        if day is None:
            chosen = next_game_day(schedule, pd.Timestamp.today()) or last_game_day(schedule)
        else:
            chosen = pd.Timestamp(day)
        if chosen is None:
            return schemas.GamesResponse(date=None, games=[])
        todays = games_on(schedule, chosen)
        if todays.empty:
            return schemas.GamesResponse(date=chosen.date(), games=[])
        todays = add_probabilities(todays, ctx.state.ratings, TUNED_ELO.home_advantage)
        # Played games: use the pre-game probability (made before the result was applied)
        pre = ctx.pre_game.set_index("game_id")["p_home"]
        has_pre = todays["game_id"].isin(pre.index) & todays["played"]
        if has_pre.any():
            todays.loc[has_pre, "p_home"] = todays.loc[has_pre, "game_id"].map(pre).to_numpy()
        names = ctx.conferences.set_index("team_id")["team"]
        sur = set(surprises(todays, 0.35)["game_id"])
        out = []
        for r in todays.itertuples():
            home, away = names.get(r.home_team_id, "TBD"), names.get(r.away_team_id, "TBD")
            out.append(
                schemas.Game(
                    game_id=r.game_id,
                    date=r.game_date.date(),
                    home_team_id=int(r.home_team_id),
                    away_team_id=int(r.away_team_id),
                    home_team=home,
                    away_team=away,
                    played=bool(r.played),
                    home_pts=int(r.home_pts) if r.played else None,
                    away_pts=int(r.away_pts) if r.played else None,
                    p_home=float(r.p_home),
                    favorite=home if r.p_home >= 0.5 else away,
                    surprise=(r.game_id in sur) if r.played else None,
                )
            )
        return schemas.GamesResponse(date=chosen.date(), games=out)

    @app.get("/v1/standings", response_model=schemas.StandingsResponse)
    def standings_endpoint(
        include_odds: bool = Query(
            default=False, description="Add playoff probabilities (runs a 3000-run simulation)"
        ),
        service: DataService = Depends(get_service),
    ):
        ctx, _ = service.context()
        table = standings(schedule_to_games(ctx.schedule, ctx.season), ctx.conferences)
        table["elo"] = table["team_id"].map(ctx.state.ratings)
        odds = service.simulation(3000, ctx=ctx)["p_playoffs"] if include_odds else None
        rows = []
        for conf, part in table.groupby("conference", sort=True):
            for rank, r in enumerate(part.itertuples(), start=1):
                rows.append(
                    schemas.Standing(
                        conference=conf,
                        rank_in_conference=rank,
                        team_id=int(r.team_id),
                        team=r.team,
                        wins=int(r.wins),
                        losses=int(r.losses),
                        win_pct=float(r.win_pct),
                        last10=r.last_n if isinstance(r.last_n, str) else "0-0",
                        point_diff=_none_if_nan(r.diff),
                        elo=_none_if_nan(r.elo),
                        p_playoffs=_odds_for(odds, r.team_id),
                    )
                )
        return schemas.StandingsResponse(
            season=ctx.season, data_as_of=_to_date(last_game_day(ctx.schedule)), standings=rows
        )

    @app.get("/v1/simulation", response_model=schemas.SimulationResponse)
    def simulation(
        n_sims: int = Query(default=5000, ge=500, le=MAX_SIMS),
        service: DataService = Depends(get_service),
    ):
        ctx, _ = service.context()
        table = service.simulation(n_sims, ctx=ctx)
        from sportslab.models.season_sim import NOTES, VALIDATED_RATING_SD

        names = ctx.conferences.set_index("team_id")
        teams = [
            schemas.SimTeam(
                team_id=int(tid),
                team=names["team"].get(tid, "TBD"),
                conference=names["conference"].get(tid, "TBD"),
                elo=float(r.elo),
                current_wins=int(r.current_wins),
                exp_wins=float(r.exp_wins),
                wins_p10=float(r.wins_p10),
                wins_p90=float(r.wins_p90),
                p_playoffs=float(r.p_playoffs),
                p_conf_finals=float(r.p_conf_finals),
                p_finals=float(r.p_finals),
                p_champion=float(r.p_champion),
            )
            for tid, r in table.sort_values("p_champion", ascending=False).iterrows()
        ]
        return schemas.SimulationResponse(
            season=ctx.season,
            n_sims=n_sims,
            rating_sd=VALIDATED_RATING_SD,
            monte_carlo_error=0.5 / n_sims**0.5,
            data_as_of=_to_date(last_game_day(ctx.schedule)),
            notes=NOTES,
            teams=teams,
        )

    def _projection_rows(table: pd.DataFrame) -> list[schemas.Projection]:
        return [
            schemas.Projection(
                player_id=int(r.player_id),
                player=r.player,
                team=r.team,
                opponent=r.opponent,
                home=bool(r.home),
                game_date=r.game_date.date(),
                games_this_season=int(r.n_prior_season),
                projection=float(r.projection),
                low50=float(r.low50),
                high50=float(r.high50),
                low80=float(r.low80),
                high80=float(r.high80),
            )
            for r in table.itertuples()
        ]

    @app.get("/v1/fantasy/projections", response_model=schemas.ProjectionsResponse)
    def fantasy_projections(
        team: str | None = Query(default=None, description="Team abbreviation, e.g. BOS"),
        limit: int = Query(default=50, ge=1, le=500),
        service: DataService = Depends(get_service),
    ):
        ctx, _ = service.context()
        table, stale = service.projections(ctx)
        if team:
            table = table[table["team"].str.upper() == team.upper()]
        table = table.sort_values("projection", ascending=False).head(limit)
        return schemas.ProjectionsResponse(
            data_as_of=_to_date(last_game_day(ctx.schedule)),
            stale=stale,
            projections=_projection_rows(table),
        )

    @app.get("/v1/fantasy/projections/{player_id}", response_model=schemas.ProjectionsResponse)
    def fantasy_projection_for(player_id: int, service: DataService = Depends(get_service)):
        ctx, _ = service.context()
        table, stale = service.projections(ctx)
        row = table[table["player_id"] == player_id]
        if row.empty:
            raise HTTPException(
                status_code=404,
                detail="No projection for this player: not active recently, or under 15 minutes.",
            )
        return schemas.ProjectionsResponse(
            data_as_of=_to_date(last_game_day(ctx.schedule)),
            stale=stale,
            projections=_projection_rows(row),
        )

    return app


def _odds_for(odds: pd.Series | None, team_id) -> float | None:
    """Playoff probability of a team, or None if odds were not requested or it was not simulated."""
    if odds is None:
        return None
    value = odds.get(team_id)
    return None if value is None or pd.isna(value) else float(value)


def _to_date(ts: pd.Timestamp | None) -> date | None:
    return None if ts is None else ts.date()


app = create_app()
