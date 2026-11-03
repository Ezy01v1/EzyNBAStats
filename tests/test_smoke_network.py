"""Smoke tests against the real stats.nba.com. Run explicitly: ``uv run pytest -m network``.

NBA.com changes or removes endpoints without notice, so each endpoint we depend on gets one.
"""

import pytest
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueGameLog

from sportslab.data.client import NbaClient
from sportslab.data.players import clean_player_stats
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
