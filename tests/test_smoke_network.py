"""Smoke tests against the real stats.nba.com. Run explicitly: ``uv run pytest -m network``.

NBA.com changes or removes endpoints without notice, so each endpoint we depend on gets one.
"""

import pytest
from nba_api.stats.endpoints import LeagueGameLog

from sportslab.data.client import NbaClient

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
