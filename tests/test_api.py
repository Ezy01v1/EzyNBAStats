from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sportslab.data import NbaClientError
from sportslab.data.schedule import clean_schedule
from sportslab.models.elo import run_elo
from sportslab.models.season_state import build_season_state
from sportslab.service.api import create_app, get_service
from sportslab.service.context import (
    TUNED_ELO,
    CurrentContext,
    DataService,
    ServiceUnavailable,
)

SEASON = "2026-27"
TEAM_IDS = list(range(1, 31))


def _conferences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": TEAM_IDS,
            "team": [f"Team{t}" for t in TEAM_IDS],
            "conference": ["East"] * 15 + ["West"] * 15,
            "division": "X",
        }
    )


def _schedule() -> pd.DataFrame:
    """Three finished days and two pending ones, written in NBA.com's raw shape."""
    rows = []
    gid = 0
    for day, status in (("10/20/2026", 3), ("10/21/2026", 3), ("10/22/2026", 3), ("12/01/2026", 1)):
        for i in range(0, 30, 2):
            gid += 1
            home, away = TEAM_IDS[i], TEAM_IDS[i + 1]
            if status == 3 and (i + gid) % 3 == 0:
                home, away = away, home
            rows.append(
                {
                    "gameId": f"002260{gid:04d}",
                    "gameDate": f"{day} 00:00:00",
                    "gameStatus": status,
                    "homeTeam_teamId": home,
                    "awayTeam_teamId": away,
                    "homeTeam_score": 110
                    if (gid % 2 == 0 and status == 3)
                    else 100
                    if status == 3
                    else 0,
                    "awayTeam_score": 100
                    if (gid % 2 == 0 and status == 3)
                    else 105
                    if status == 3
                    else 0,
                    "isNeutral": False,
                }
            )
    return clean_schedule(pd.DataFrame(rows))


def _history() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(300):
        a, b = rng.choice(TEAM_IDS, 2, replace=False)
        rows.append(
            {
                "season": "2025-26",
                "game_id": f"h{g}",
                "game_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=g // 8),
                "home_team_id": int(a),
                "away_team_id": int(b),
                "home_win": int(rng.random() < 0.55),
                "neutral_site": False,
            }
        )
    return pd.DataFrame(rows)


def _context(_client=None) -> CurrentContext:
    history, schedule, conf = _history(), _schedule(), _conferences()
    state = build_season_state(history, schedule, conf, SEASON, TUNED_ELO)
    from sportslab.data.schedule import schedule_to_games

    pre = run_elo(
        pd.concat([history, schedule_to_games(schedule, SEASON)], ignore_index=True), TUNED_ELO
    )
    return CurrentContext(SEASON, schedule, conf, state, pre, loaded_at=datetime(2026, 11, 3, 9))


class FakeService(DataService):
    """Real context logic on synthetic data; projections without any network."""

    def projections(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "player": ["A", "B", "C"],
                "team": ["BOS", "BOS", "NYK"],
                "opponent": ["NYK", "NYK", "BOS"],
                "home": [True, True, False],
                "game_date": pd.to_datetime(["2026-12-01"] * 3),
                "n_prior_season": [8, 8, 5],
                "season_mean": [30.0, 20.0, 25.0],
                "projection": [31.0, 22.0, 26.0],
                "low50": [24.0, 16.0, 19.0],
                "high50": [38.0, 28.0, 33.0],
                "low80": [18.0, 10.0, 12.0],
                "high80": [45.0, 34.0, 40.0],
            }
        )


@pytest.fixture
def client():
    service = FakeService(client=object(), context_loader=_context)
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app), service


def test_health_does_not_load_any_data():
    calls = []
    service = DataService(client=object(), context_loader=lambda c: calls.append(1) or _context())
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    assert TestClient(app).get("/health").json() == {"status": "ok"}
    assert calls == []


def test_meta_reports_the_season_data_date_and_the_schedule_warning(client):
    http, _ = client
    body = http.get("/v1/meta").json()
    assert body["season"] == SEASON and body["stale"] is False
    assert body["data_as_of"] == "2026-10-22"
    assert body["elo"]["home_advantage"] == 40
    assert any("instead of 82" in w for w in body["warnings"])
    assert "betting advice" in body["disclaimer"]


def test_ratings_cover_every_team_sorted_by_elo(client):
    http, _ = client
    rows = http.get("/v1/ratings").json()
    assert len(rows) == 30 and {r["conference"] for r in rows} == {"East", "West"}
    elos = [r["elo"] for r in rows]
    assert elos == sorted(elos, reverse=True)


def test_games_default_to_the_next_day_with_pregame_probabilities(client):
    http, _ = client
    body = http.get("/v1/games").json()
    assert body["date"] == "2026-12-01" and len(body["games"]) == 15
    g = body["games"][0]
    assert g["played"] is False and g["home_pts"] is None and g["surprise"] is None
    assert 0 < g["p_home"] < 1 and g["favorite"] in (g["home_team"], g["away_team"])


def test_a_finished_day_has_scores_and_surprise_flags(client):
    http, _ = client
    body = http.get("/v1/games", params={"date": "2026-10-21"}).json()
    assert len(body["games"]) == 15 and all(g["played"] for g in body["games"])
    assert all(isinstance(g["surprise"], bool) and g["home_pts"] is not None for g in body["games"])


def test_a_day_without_games_is_empty_and_a_bad_date_is_rejected(client):
    http, _ = client
    assert http.get("/v1/games", params={"date": "2026-11-15"}).json()["games"] == []
    assert http.get("/v1/games", params={"date": "not-a-date"}).status_code == 422


def test_standings_rank_inside_each_conference_and_odds_are_optional(client):
    http, _ = client
    plain = http.get("/v1/standings").json()
    assert len(plain["standings"]) == 30
    assert all(s["p_playoffs"] is None for s in plain["standings"])
    east = [s for s in plain["standings"] if s["conference"] == "East"]
    assert [s["rank_in_conference"] for s in east] == list(range(1, 16))
    with_odds = http.get("/v1/standings", params={"include_odds": "true"}).json()
    assert sum(s["p_playoffs"] for s in with_odds["standings"]) == pytest.approx(16, abs=1e-6)


def test_simulation_validates_its_size_and_returns_probabilities_that_add_up(client):
    http, _ = client
    assert http.get("/v1/simulation", params={"n_sims": 10}).status_code == 422
    assert http.get("/v1/simulation", params={"n_sims": 999999}).status_code == 422
    body = http.get("/v1/simulation", params={"n_sims": 800}).json()
    assert body["n_sims"] == 800 and body["rating_sd"] == 75.0
    assert sum(t["p_champion"] for t in body["teams"]) == pytest.approx(1.0)
    champs = [t["p_champion"] for t in body["teams"]]
    assert champs == sorted(champs, reverse=True)
    assert body["monte_carlo_error"] == pytest.approx(0.5 / 800**0.5)


def test_projections_filter_by_team_limit_and_player(client):
    http, _ = client
    everything = http.get("/v1/fantasy/projections").json()
    assert [p["player"] for p in everything["projections"]] == ["A", "C", "B"]  # by projection
    assert "if" in everything["note"].lower() or "conditional" in everything["note"].lower()
    bos = http.get("/v1/fantasy/projections", params={"team": "bos"}).json()
    assert {p["team"] for p in bos["projections"]} == {"BOS"}
    assert len(http.get("/v1/fantasy/projections", params={"limit": 1}).json()["projections"]) == 1
    one = http.get("/v1/fantasy/projections/2").json()["projections"][0]
    assert one["low80"] < one["low50"] < one["projection"] < one["high50"] < one["high80"]
    assert http.get("/v1/fantasy/projections/999").status_code == 404


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_the_context_is_cached_until_the_ttl_and_refreshed_after_it():
    clock, calls = _Clock(), []
    service = DataService(
        client=object(),
        context_loader=lambda c: calls.append(1) or _context(),
        ttl=timedelta(minutes=30),
        clock=clock,
    )
    service.context()
    service.context()
    assert len(calls) == 1
    clock.t = 31 * 60
    service.context()
    assert len(calls) == 2


def test_stale_data_is_served_when_a_refresh_fails_and_503_without_any_data():
    clock = _Clock()
    state = {"fail": False}

    def loader(_client):
        if state["fail"]:
            raise NbaClientError("blocked")
        return _context()

    service = FakeService(
        client=object(), context_loader=loader, ttl=timedelta(minutes=1), clock=clock
    )
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    http = TestClient(app)
    assert http.get("/v1/meta").json()["stale"] is False
    clock.t, state["fail"] = 120.0, True
    assert http.get("/v1/meta").json()["stale"] is True  # old copy, flagged

    empty = FakeService(client=object(), context_loader=loader, clock=_Clock())
    app2 = create_app()
    app2.dependency_overrides[get_service] = lambda: empty
    response = TestClient(app2).get("/v1/meta")
    assert response.status_code == 503 and "no cached copy" in response.json()["detail"]
    with pytest.raises(ServiceUnavailable):
        empty.context()
    assert date.today()  # sanity: the module imports date for the meta payload
