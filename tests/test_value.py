import numpy as np
import pandas as pd
import pytest

from sportslab.analysis.value import (
    contribution,
    fit_market,
    gap_persistence,
    market_fit_summary,
    score_players,
)
from sportslab.data.salaries import (
    match_salaries,
    normalize_name,
    parse_salary,
    read_salaries,
)


def test_names_are_normalized_across_accents_punctuation_and_suffixes():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Jaren Jackson Jr.") == normalize_name("Jaren Jackson")
    assert normalize_name("P.J. Washington") == normalize_name("PJ Washington") == "pj washington"
    assert normalize_name("De'Aaron Fox") == "deaaron fox"
    assert normalize_name("Nicolas Claxton") == normalize_name("Nic Claxton")
    assert normalize_name("Dennis Schroeder") == normalize_name("Dennis Schröder")
    assert normalize_name("Gary Trent  Jr") == "gary trent"


def test_salary_values_are_parsed_or_flagged():
    assert parse_salary(51415938) == 51415938
    assert parse_salary("$51,415,938") == 51415938
    assert np.isnan(parse_salary("n/a"))


def _csv(tmp_path, text: str):
    path = tmp_path / "salaries.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_reading_a_valid_csv_adds_a_name_key(tmp_path):
    path = _csv(tmp_path, "season,player_name,team,salary_usd\n2025-26,Nikola Jokić,DEN,51415938\n")
    out = read_salaries(path)
    assert out.loc[0, "name_key"] == "nikola jokic" and out.loc[0, "salary_usd"] == 51415938


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("season,player_name\n2025-26,A\n", "missing columns"),
        ("season,player_name,salary_usd\n2025-26,A,abc\n", "salary"),
        ("season,player_name,salary_usd\n2025-26,A,0\n", "salary"),
        ("season,player_name,salary_usd\n2025,A,100\n", "season must look like"),
        ("season,player_name,salary_usd\n2025-26,A,100\n2025-26,A,200\n", "duplicate"),
    ],
)
def test_a_bad_csv_fails_with_a_clear_message(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_salaries(_csv(tmp_path, text))


def _players(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pie = rng.uniform(0.05, 0.2, n)
    minutes = rng.uniform(1100, 2800, n)
    return pd.DataFrame(
        {
            "season": "2025-26",
            "player_id": range(n),
            "player": [f"Player {i}" for i in range(n)],
            "team": "AAA",
            "age": rng.integers(20, 36, n),
            "pie": pie,
            "minutes": minutes,
        }
    )


def _with_salary(players: pd.DataFrame, seed: int = 1, underpaid=()) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = players.copy()
    log_salary = 16.5 + 0.9 * np.log(contribution(out)) / 1 + rng.normal(0, 0.1, len(out))
    out["salary_usd"] = np.exp(log_salary - 0.9 * np.log(contribution(out)).mean() + 16.0)
    for i in underpaid:
        out.loc[i, "salary_usd"] *= 0.3
    return out


def test_match_salaries_reports_every_unmatched_name_on_both_sides(tmp_path):
    players = pd.DataFrame(
        {
            "season": ["2025-26"] * 3,
            "player_id": [1, 2, 3],
            "player": ["Luka Dončić", "Jaren Jackson Jr.", "Nobody Here"],
            "team": ["DAL", "MEM", "XXX"],
        }
    )
    path = _csv(
        tmp_path,
        "season,player_name,salary_usd\n2025-26,Luka Doncic,45000000\n2025-26,Jaren Jackson,"
        "30000000\n2025-26,Ghost Player,1000000\n",
    )
    matched, csv_left, players_left = match_salaries(read_salaries(path), players)
    assert set(matched["player"]) == {"Luka Dončić", "Jaren Jackson Jr."}
    assert csv_left["player_name"].tolist() == ["Ghost Player"]
    assert players_left["player"].tolist() == ["Nobody Here"]


def test_traded_players_salary_rows_are_added_up(tmp_path):
    players = pd.DataFrame(
        {"season": ["2025-26"], "player_id": [1], "player": ["A Player"], "team": ["BOS"]}
    )
    path = _csv(
        tmp_path,
        "season,player_name,team,salary_usd\n2025-26,A Player,BOS,10000000\n"
        "2025-26,A Player,NYK,5000000\n",
    )
    matched, _, _ = match_salaries(read_salaries(path), players)
    assert matched.loc[0, "salary_usd"] == 15_000_000


def test_planted_underpaid_players_get_the_lowest_ratios_with_intervals_below_one():
    players = _with_salary(_players(), underpaid=[3, 17])
    scored = score_players(players, rookie_control=False, n_boot=200)
    worst = scored.nsmallest(2, "ratio")
    assert set(worst["player_id"]) == {3, 17}
    assert (worst["ratio"] < 0.5).all() and (worst["ratio_high"] < 1).all()
    assert (scored["ratio_low"] <= scored["ratio"]).all()
    assert (scored["ratio"] <= scored["ratio_high"]).all()


def test_rotation_filter_and_fit_summary_recover_the_planted_elasticity():
    players = _with_salary(_players(n=200, seed=3))
    players.loc[0, "minutes"] = 200  # not a rotation player
    assert 0 not in score_players(players, rookie_control=False, n_boot=10)["player_id"].tolist()
    summary = market_fit_summary(players, rookie_control=False)
    assert summary["elasticity"] == pytest.approx(0.9, abs=0.1) and summary["r2"] > 0.9
    assert fit_market(players, rookie_control=False).coef_[0] == pytest.approx(0.9, abs=0.1)


def test_gap_persistence_is_high_when_underpayment_is_a_lasting_trait():
    base = _players(n=150, seed=5)
    rng = np.random.default_rng(7)
    trait = rng.normal(0, 0.3, len(base))  # a lasting over/underpayment per player

    def season(seed):
        df = _with_salary(base, seed=seed)
        df["salary_usd"] *= np.exp(trait)
        return df

    rho, low, high, n = gap_persistence(season(1), season(2), rookie_control=False, n_boot=100)
    assert n == 150 and low > 0.5 and low <= rho <= high


def test_gap_z_flags_only_players_far_from_the_usual_spread():
    players = _with_salary(_players(n=200, seed=11), underpaid=[5])
    scored = score_players(players, rookie_control=False, n_boot=10).set_index("player_id")
    assert scored.loc[5, "gap_z"] < -2 and abs(scored["gap_z"].median()) < 0.3
    assert (scored["gap_z"].abs() > 2).sum() <= 6  # ~5% by chance, plus the planted one


def test_experience_counts_seasons_since_the_first_one():
    from sportslab.data.players import add_experience

    players = pd.DataFrame({"season": ["2022-23"] * 3, "player_id": [1, 2, 3]})
    career = pd.DataFrame({"player_id": [1, 2], "from_year": [2022, 2013]})
    out = add_experience(players, career).set_index("player_id")
    assert out.loc[1, "years_experience"] == 0 and out.loc[2, "years_experience"] == 9
    assert pd.isna(out.loc[3, "years_experience"])
