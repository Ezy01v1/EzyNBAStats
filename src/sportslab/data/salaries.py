"""Salaries come from a CSV YOU provide (NBA.com has no salary data, and we do not scrape sites
whose terms forbid it). Expected file, e.g. ``data/external/salaries.csv``::

    season,player_name,team,salary_usd
    2025-26,Nikola Jokić,DEN,51415938

Required: ``season`` (2025-26 style), ``player_name``, ``salary_usd`` (whole dollars).
Optional: ``team`` (abbreviation, only used to break name ties), ``player_id`` (NBA.com id, which
skips name matching) and ``years_experience`` (to control for rookie-scale contracts).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

REQUIRED = ["season", "player_name", "salary_usd"]
OPTIONAL = ["team", "player_id", "years_experience"]
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# Same player, different spelling in the two sources (found by checking who failed to match).
ALIASES = {
    "dennis schroeder": "dennis schroder",
    "herb jones": "herbert jones",
    "nicolas claxton": "nic claxton",
    "santiago aldama": "santi aldama",
}


def normalize_name(name: str) -> str:
    """Lowercase, no accents, periods or apostrophes, no Jr./III suffix, known aliases applied:
    "Luka Dončić" -> "luka doncic", "P.J. Washington" -> "pj washington"."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[.'’]", "", text)  # "p.j." and "de'aaron" collapse instead of splitting
    tokens = re.sub(r"[^a-z0-9 ]", " ", text).split()
    key = " ".join(t for t in tokens if t not in SUFFIXES)
    return ALIASES.get(key, key)


def parse_salary(value) -> float:
    """Accepts 51415938, "51,415,938" or "$51,415,938"; anything else is NaN (and reported)."""
    if isinstance(value, int | float):
        return float(value)
    cleaned = re.sub(r"[,$\s]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def read_salaries(path: str | Path) -> pd.DataFrame:
    """Read and validate the CSV. Raises with a clear message if the schema is wrong."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(
            f"salaries CSV is missing columns {missing}; expected at least {REQUIRED} "
            f"(optional: {OPTIONAL})"
        )
    out = raw.copy()
    out["salary_usd"] = out["salary_usd"].map(parse_salary)
    bad = out["salary_usd"].isna() | (out["salary_usd"] <= 0)
    if bad.any():
        examples = out.loc[bad, "player_name"].head(5).tolist()
        raise ValueError(f"{int(bad.sum())} rows have a missing or non-positive salary: {examples}")
    if not out["season"].str.match(r"^\d{4}-\d{2}$").all():
        raise ValueError("season must look like 2025-26")
    for column in ("player_id", "years_experience"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["name_key"] = out["player_name"].map(normalize_name)
    key = ["season", "name_key"] + (["team"] if "team" in out else [])
    if out.duplicated(key).any():
        raise ValueError(
            "duplicate (season, player" + (", team" if "team" in out else "") + ") rows"
        )
    return out


def match_salaries(
    salaries: pd.DataFrame, players: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach salaries to NBA.com players (``player_id``, ``player``, ``season``, ``team``).

    Matches on ``player_id`` when the CSV has it, otherwise on (season, normalized name). A player
    traded mid-season may have one salary row per team: those are summed before matching.
    Returns (matched, unmatched_in_csv, unmatched_in_players) so nothing disappears silently.
    """
    sal = salaries.copy()
    if "player_id" in sal and sal["player_id"].notna().all():
        agg = sal.groupby(["season", "player_id"], as_index=False)["salary_usd"].sum()
        merged = players.merge(agg, on=["season", "player_id"], how="left")
        csv_unmatched = sal[~sal["player_id"].isin(players["player_id"])]
    else:
        sal_sum = sal.groupby(["season", "name_key"], as_index=False)["salary_usd"].sum()
        extra = [c for c in ("years_experience",) if c in sal]
        if extra:
            sal_sum = sal_sum.merge(
                sal.groupby(["season", "name_key"], as_index=False)[extra].first(),
                on=["season", "name_key"],
            )
        pl = players.assign(name_key=players["player"].map(normalize_name))
        merged = pl.merge(sal_sum, on=["season", "name_key"], how="left")
        keys = set(zip(pl["season"], pl["name_key"], strict=True))
        csv_unmatched = sal[
            [(s, k) not in keys for s, k in zip(sal["season"], sal["name_key"], strict=True)]
        ]
    matched = merged[merged["salary_usd"].notna()].reset_index(drop=True)
    players_unmatched = merged[merged["salary_usd"].isna()].reset_index(drop=True)
    return matched, csv_unmatched.reset_index(drop=True), players_unmatched
