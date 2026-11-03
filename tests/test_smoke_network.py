"""Smoke tests against the real stats.nba.com. Run explicitly: ``uv run pytest -m network``.

NBA.com changes or removes endpoints without notice, so each endpoint we depend on gets one.
"""

import pytest
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueGameLog

from sportslab.data.awards import award_labels, load_player_awards
from sportslab.data.client import NbaClient
from sportslab.data.lineups import load_lineups, load_team_ratings
from sportslab.data.player_season import load_player_season
from sportslab.data.players import clean_player_stats
from sportslab.data.profiles import load_per100, load_shot_zones
from sportslab.data.schedule import (
    games_per_team,
    load_conferences,
    load_playoff_games,
    load_schedule,
    playoff_outcomes,
)
from sportslab.data.shots import load_player_shots

pytestmark = pytest.mark.network


def test_league_game_log_regular_season_2023_24(tmp_path):
    client = NbaClient(cache_dir=tmp_path, timeout=60)
    (games,) = client.fetch(LeagueGameLog, season="2023-24", season_type_all_star="Regular Season")
    # 30 teams x 82 games = 2460 team-game rows (each game appears once per team)
    assert len(games) == 2460
    assert {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP", "WL", "PTS"} <= set(
        games.columns
    )
    assert games["GAME_ID"].nunique() == 1230


def test_league_dash_player_stats_advanced_2023_24(tmp_path):
    client = NbaClient(cache_dir=tmp_path, timeout=60)
    (players,) = client.fetch(
        LeagueDashPlayerStats,
        season="2023-24",
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
    )
    assert {"PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "TS_PCT", "USG_PCT", "PIE", "POSS"} <= set(
        players.columns
    )
    assert players["PLAYER_ID"].is_unique and len(players) > 400
    # clean_player_stats raises if MIN stops being per-game (possessions-per-minute check)
    assert clean_player_stats(players, "2023-24")["minutes"].max() > 2000


def test_shot_chart_detail_for_one_player_season(tmp_path):
    client = NbaClient(cache_dir=tmp_path, timeout=60)
    shots, league = load_player_shots(client, player_id=203999, season="2023-24")  # Jokic
    assert 800 < len(shots) < 1600 and shots["made"].isin([0, 1]).all()
    assert shots["points"].isin([2, 3]).all()
    # the league zone totals cover every shot of the season: ~200k attempts
    assert 150_000 < league["fga"].sum() < 260_000
    assert {"Restricted Area", "Mid-Range", "Left Corner 3"} <= set(league["zone_basic"])


def test_player_style_tables_cover_the_same_players(tmp_path):
    client = NbaClient(cache_dir=tmp_path, timeout=60)
    per100 = load_per100(client, "2023-24")
    zones = load_shot_zones(client, "2023-24")
    assert len(per100) > 400 and per100["player_id"].is_unique
    assert set(zones["player_id"]) <= set(per100["player_id"])
    assert (zones[["ra_fga", "mid_fga", "atb3_fga"]] >= 0).all().all()
    assert 40 < per100["fga100"].median() * 3 < 100  # sanity: ~per-100 scale, not per game


def test_lineups_are_capped_and_team_ratings_cover_30_teams(tmp_path):
    from datetime import date

    client = NbaClient(cache_dir=tmp_path, timeout=60)
    five = load_lineups(client, "2023-24", 5)
    # NBA.com returns at most ~2000 lineups (the busiest ones); this limit shapes every analysis
    assert 1500 < len(five) <= 2000 and five["minutes"].is_monotonic_decreasing
    half = load_lineups(client, "2023-24", 2, date_to=date(2024, 1, 15))
    assert (half["poss"] > 0).all() and not half.duplicated(["lineup_id", "team_id"]).any()
    teams = load_team_ratings(client, "2023-24")
    assert len(teams) == 30 and teams["net_rating"].abs().max() < 20


def test_schedule_conferences_and_playoff_results_for_a_finished_season(tmp_path):
    client = NbaClient(cache_dir=tmp_path, timeout=60)
    schedule = load_schedule(client, "2023-24")
    assert len(schedule) == 1230 and schedule["played"].all()
    assert (games_per_team(schedule) == 82).all()
    conferences = load_conferences(client, "2023-24")
    assert conferences["conference"].value_counts().to_dict() == {"East": 15, "West": 15}
    outcome = playoff_outcomes(load_playoff_games(client, "2023-24"))
    assert len(outcome["made_playoffs"]) == 16 and len(outcome["finalists"]) == 2
    boston = int(conferences.loc[conferences["team"] == "Celtics", "team_id"].iloc[0])
    assert outcome["champion"] == boston  # the 2023-24 champions


def test_awards_and_cutoff_player_stats(tmp_path):
    from datetime import date

    client = NbaClient(cache_dir=tmp_path, timeout=60)
    awards = load_player_awards(client, 203999)  # Jokic: MVP 2020-21, 2021-22, 2023-24
    labels = award_labels(awards).set_index("season")
    assert bool(labels.loc["2020-21", "mvp"]) and bool(labels.loc["2023-24", "all_star"])
    stats = load_player_season(client, "2023-24", date(2024, 1, 15))
    assert len(stats) > 400 and stats["team_w_pct"].notna().all()
    assert stats["gp"].max() < 50  # through mid-January, not the whole season
    assert 25 < stats["pts"].max() < 40
