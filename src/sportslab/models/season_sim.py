"""Monte Carlo season simulator: remaining games -> standings -> play-in -> playoffs -> champion.

Every simulated season draws each remaining game from the Elo win probability, then ranks teams
inside each conference, plays the play-in (seeds 7-10), and runs a fixed bracket of best-of-7
series. Simplifications, all deliberate and listed in ``SeasonSimulation.notes``:

- Ratings are frozen at the cutoff (they are not updated by simulated results), but each
  simulated season draws every team's true strength around its rating with ``rating_sd``:
  Elo is an estimate, not the truth, and ignoring that makes every interval too narrow.
- Ties in wins are broken at random (the real tiebreak rules are long).
- Home court in a series goes to the team with the better simulated record (random tiebreak).
- Injuries, rest and lineups are not modeled: Elo only sees past results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOME_PATTERN = (True, True, False, False, True, False, True)  # 2-2-1-1-1: higher seed at home
BRACKET = ((1, 8), (4, 5), (3, 6), (2, 7))  # first-round matchups by seed (fixed bracket)
NOTES = (
    "Ratings frozen at the cutoff; ties in wins broken at random; home court by simulated record; "
    "no injuries, rest or lineups; unscheduled games are played against an average opponent."
)


# Chosen by leave-one-season-out in scripts/season_backtest.py (2021-22..2025-26): with frozen
# ratings the 80% win interval covered 59% of teams; at 75 it covers 83% (held-out: 78-88%).
VALIDATED_RATING_SD = 75.0


@dataclass(frozen=True)
class SimConfig:
    home_advantage: float = 40.0  # Elo points, the value tuned in Module 1
    n_sims: int = 10_000
    seed: int = 0
    initial: float = 1500.0
    rating_sd: float = VALIDATED_RATING_SD  # Elo points of uncertainty about true team strength


@dataclass
class SeasonSimulation:
    table: pd.DataFrame  # one row per team, see ``simulate_season``
    n_sims: int
    notes: str = NOTES
    extra: dict = field(default_factory=dict)

    def monte_carlo_error(self) -> float:
        """Largest standard error of any probability in the table (worst case p = 0.5)."""
        return float(0.5 / np.sqrt(self.n_sims))


def win_prob(r_home, r_away, home_advantage):
    """Elo probability that the home side wins (works with arrays)."""
    return 1.0 / (1.0 + 10.0 ** (-(np.asarray(r_home) + home_advantage - np.asarray(r_away)) / 400))


def series_win_prob(p_home, p_away):
    """Probability that the higher seed wins a best-of-7 (2-2-1-1-1), given its win probability
    in a game at home (``p_home``) and on the road (``p_away``). Exact, vectorized over arrays."""
    p_home, p_away = np.asarray(p_home, dtype=float), np.asarray(p_away, dtype=float)
    states = {(0, 0): np.ones_like(p_home)}
    won = np.zeros_like(p_home)
    for at_home in HOME_PATTERN:
        p = p_home if at_home else p_away
        nxt: dict[tuple[int, int], np.ndarray] = {}
        for (a, b), prob in states.items():
            for da, weight in ((1, p), (0, 1 - p)):
                state = (a + da, b + 1 - da)
                mass = prob * weight
                if state[0] == 4:
                    won = won + mass
                elif state[1] == 4:
                    continue
                else:
                    nxt[state] = nxt.get(state, 0) + mass
        states = nxt
    return won


def _r(rating: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Rating of team ``idx[i]`` in simulation i (``rating`` is (n_sims, n_teams))."""
    return rating[np.arange(len(idx)), idx]


def _series(
    a: np.ndarray, b: np.ndarray, score: np.ndarray, rating: np.ndarray, h: float, rng
) -> np.ndarray:
    """Winners (team indices) of series between arrays of teams ``a`` and ``b``; the team with
    the higher ``score`` (record + random tiebreak) has home court."""
    a_has_court = score[np.arange(len(a)), a] >= score[np.arange(len(b)), b]
    hc, other = np.where(a_has_court, a, b), np.where(a_has_court, b, a)
    p_home = win_prob(_r(rating, hc), _r(rating, other), h)
    p_away = 1.0 - win_prob(_r(rating, other), _r(rating, hc), h)
    hc_wins = rng.random(len(a)) < series_win_prob(p_home, p_away)
    return np.where(hc_wins, hc, other)


def _play_in(seeds: np.ndarray, rating: np.ndarray, h: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Seeds 7 and 8 from the four play-in teams (``seeds[:, 6:10]``)."""
    s7, s8, s9, s10 = (seeds[:, i] for i in range(6, 10))
    n = len(s7)
    first = rng.random(n) < win_prob(_r(rating, s7), _r(rating, s8), h)  # 7 hosts 8
    winner_a, loser_a = np.where(first, s7, s8), np.where(first, s8, s7)
    second = rng.random(n) < win_prob(_r(rating, s9), _r(rating, s10), h)  # 9 hosts 10
    winner_b = np.where(second, s9, s10)
    last = rng.random(n) < win_prob(_r(rating, loser_a), _r(rating, winner_b), h)  # loser hosts
    return winner_a, np.where(last, loser_a, winner_b)


def simulate_season(
    ratings: Mapping[int, float],
    conferences: pd.DataFrame,
    current_wins: Mapping[int, int],
    remaining: pd.DataFrame,
    extra_games: Mapping[int, int] | None = None,
    config: SimConfig | None = None,
) -> SeasonSimulation:
    """Simulate the rest of a season many times.

    ``conferences``: team_id, conference (15 teams each). ``remaining``: home_team_id,
    away_team_id, neutral_site (a team id missing from ``conferences``, e.g. 0 for "to be
    determined", is an average-rated opponent that is not ranked). ``extra_games``: games a team
    still owes that are not in ``remaining`` (played against an average opponent at a neutral
    site). Returns a table with, per team: current and expected wins with percentiles, and the
    probabilities of top-6, play-in, playoffs, second round, conference finals, finals, title.
    """
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    teams = conferences["team_id"].to_numpy()
    t_index = {t: i for i, t in enumerate(teams)}
    n_teams, n = len(teams), cfg.n_sims
    rating = np.array([ratings.get(t, cfg.initial) for t in teams], dtype=float)
    mean_rating = float(rating.mean())
    # True strength of every team in every simulated season: the estimate plus its uncertainty.
    sim_rating = rating + cfg.rating_sd * rng.standard_normal((n, n_teams))
    padded = np.column_stack([sim_rating, np.full(n, cfg.initial)])  # last column: unknown team

    # --- remaining regular season -------------------------------------------------------
    unknown = n_teams
    home = remaining["home_team_id"].to_numpy()
    away = remaining["away_team_id"].to_numpy()
    home_col = np.array([t_index.get(t, unknown) for t in home], dtype=int)
    away_col = np.array([t_index.get(t, unknown) for t in away], dtype=int)
    h = np.where(remaining["neutral_site"].to_numpy(), 0.0, cfg.home_advantage)
    p_game = win_prob(padded[:, home_col], padded[:, away_col], h)  # (n, games)
    home_wins = rng.random(p_game.shape) < p_game

    to_home = np.zeros((len(home), n_teams), dtype=np.float32)
    to_away = np.zeros_like(to_home)
    for g, (hc, ac) in enumerate(zip(home_col, away_col, strict=True)):
        if hc < unknown:
            to_home[g, hc] = 1.0
        if ac < unknown:
            to_away[g, ac] = 1.0
    base = np.array([current_wins.get(t, 0) for t in teams], dtype=float)
    wins = base + home_wins.astype(np.float32) @ to_home + (~home_wins).astype(np.float32) @ to_away
    if extra_games:
        p_avg = win_prob(sim_rating, mean_rating, 0.0)  # (n, teams)
        owed = np.array([extra_games.get(t, 0) for t in teams])
        wins = wins + rng.binomial(owed, p_avg)

    # --- seeding, play-in, bracket ----------------------------------------------------------
    score = wins + rng.random((n, n_teams))  # random tiebreak: wins differ by whole numbers
    stage = {
        k: np.zeros(n_teams) for k in ("top6", "playin", "playoffs", "r2", "cf", "finals", "title")
    }
    champions = {}
    for conf, members in conferences.groupby("conference")["team_id"]:
        idx = np.array([t_index[t] for t in members])
        order = idx[np.argsort(-score[:, idx], axis=1)]  # (n, 15): team index by seed
        seven, eight = _play_in(order, sim_rating, cfg.home_advantage, rng)
        seed_team = {s: order[:, s - 1] for s in range(1, 7)} | {7: seven, 8: eight}
        for s in range(6):
            stage["top6"] += np.bincount(order[:, s], minlength=n_teams)
        for s in range(6, 10):
            stage["playin"] += np.bincount(order[:, s], minlength=n_teams)
        for s in range(1, 9):
            stage["playoffs"] += np.bincount(seed_team[s], minlength=n_teams)
        r1 = [
            _series(seed_team[a], seed_team[b], score, sim_rating, cfg.home_advantage, rng)
            for a, b in BRACKET
        ]
        for w in r1:
            stage["r2"] += np.bincount(w, minlength=n_teams)
        semis = [
            _series(r1[0], r1[1], score, sim_rating, cfg.home_advantage, rng),
            _series(r1[2], r1[3], score, sim_rating, cfg.home_advantage, rng),
        ]
        for w in semis:
            stage["cf"] += np.bincount(w, minlength=n_teams)
        champions[conf] = _series(semis[0], semis[1], score, sim_rating, cfg.home_advantage, rng)
        stage["finals"] += np.bincount(champions[conf], minlength=n_teams)
    east_west = list(champions.values())
    winner = _series(east_west[0], east_west[1], score, sim_rating, cfg.home_advantage, rng)
    stage["title"] += np.bincount(winner, minlength=n_teams)

    table = pd.DataFrame(
        {
            "elo": rating,
            "current_wins": base,
            "exp_wins": wins.mean(axis=0),
            **{
                f"wins_p{int(q * 100):02d}": np.quantile(wins, q, axis=0)
                for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
            },
            "p_top6": stage["top6"] / n,
            "p_playin": stage["playin"] / n,
            "p_playoffs": stage["playoffs"] / n,
            "p_round2": stage["r2"] / n,
            "p_conf_finals": stage["cf"] / n,
            "p_finals": stage["finals"] / n,
            "p_champion": stage["title"] / n,
        },
        index=pd.Index(teams, name="team_id"),
    )
    return SeasonSimulation(table=table, n_sims=n)
