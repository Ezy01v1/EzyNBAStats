import pandas as pd

from sportslab.data.games import GAME_COLUMNS, team_logs_to_games


def _log() -> pd.DataFrame:
    # Two games, one row per team (the shape LeagueGameLog returns). Deliberately unsorted.
    return pd.DataFrame(
        {
            "GAME_ID": ["002", "001", "002", "001"],
            "TEAM_ID": [20, 10, 10, 20],
            "TEAM_ABBREVIATION": ["NYK", "BOS", "BOS", "NYK"],
            "GAME_DATE": ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02"],
            "MATCHUP": ["NYK vs. BOS", "BOS vs. NYK", "BOS @ NYK", "NYK @ BOS"],
            "WL": ["W", "W", "L", "L"],
            "PTS": [100, 110, 90, 105],
        }
    )


def test_one_row_per_game_sorted_by_date():
    games = team_logs_to_games(_log(), season="2023-24")
    assert list(games["game_id"]) == ["001", "002"]
    assert games["game_date"].is_monotonic_increasing


def test_home_and_away_come_from_matchup_text():
    games = team_logs_to_games(_log(), season="2023-24").set_index("game_id")
    assert games.loc["001", "home_team_id"] == 10 and games.loc["001", "away_team_id"] == 20
    assert games.loc["002", "home_team_id"] == 20 and games.loc["002", "away_team_id"] == 10


def test_scores_and_home_win():
    games = team_logs_to_games(_log(), season="2023-24").set_index("game_id")
    assert (games.loc["001", "home_pts"], games.loc["001", "away_pts"]) == (110, 105)
    assert games.loc["001", "home_win"] == 1  # BOS (home) won game 001
    assert games.loc["002", "home_win"] == 1  # NYK (home) won game 002


def test_neutral_site_game_has_no_real_home_team():
    # Both rows say "@" (e.g. a game played in Mexico City): there is no home team.
    log = pd.concat(
        [
            _log(),
            pd.DataFrame(
                {
                    "GAME_ID": ["003", "003"],
                    "TEAM_ID": [30, 40],
                    "TEAM_ABBREVIATION": ["MIA", "WAS"],
                    "GAME_DATE": ["2024-01-04", "2024-01-04"],
                    "MATCHUP": ["MIA @ WAS", "WAS @ MIA"],
                    "WL": ["W", "L"],
                    "PTS": [118, 98],
                }
            ),
        ],
        ignore_index=True,
    )
    games = team_logs_to_games(log, season="2023-24").set_index("game_id")
    assert list(games["neutral_site"]) == [False, False, True]
    assert games.loc["003", "home_win"] in (0, 1)  # label is arbitrary, but row is kept
    assert games.loc["003", "home_pts"] + games.loc["003", "away_pts"] == 216


def test_season_label_is_attached():
    assert set(team_logs_to_games(_log(), season="2023-24")["season"]) == {"2023-24"}


def test_empty_log_gives_empty_games_with_expected_columns():
    # e.g. a season whose first game has not been played yet
    games = team_logs_to_games(_log().iloc[0:0], season="2026-27")
    assert games.empty
    assert list(games.columns) == GAME_COLUMNS
