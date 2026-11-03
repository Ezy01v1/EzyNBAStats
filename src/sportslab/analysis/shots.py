"""Zone efficiency of a shooter against the league, with sample-size honesty.

Free throws are not included: "points per shot" here is 2 or 3 times the field-goal rate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sportslab.models.metrics import wilson_interval

THREE_POINT_ZONES = {"Above the Break 3", "Left Corner 3", "Right Corner 3"}
IGNORED_ZONES = {"Backcourt"}  # ~30 heaves per league season: no information
BASIC = ("zone_basic",)
BASIC_RANGE = ("zone_basic", "zone_range")
# Pseudo-attempts at the league rate used for shrinkage. Chosen with scripts/shot_shrinkage.py:
# predicting 2nd-half zone rates from 1st-half ones for 100 players (2025-26), the error is flat
# between k=80 and k=320 (minimum near 160) and 40% below the raw rate; 20 was clearly too weak.
PRIOR_ATTEMPTS = 100


def zone_value(zone_basic: str) -> int:
    """Points of a made shot from this zone."""
    return 3 if zone_basic in THREE_POINT_ZONES else 2


def _keep(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame["zone_basic"].isin(IGNORED_ZONES)]


def league_zone_table(league: pd.DataFrame, keys: Sequence[str] = BASIC) -> pd.DataFrame:
    """League attempts, field-goal rate, share of all shots and points per shot by zone."""
    grouped = _keep(league).groupby(list(keys))[["fga", "fgm"]].sum()
    grouped["fg_pct"] = grouped["fgm"] / grouped["fga"]
    grouped["share"] = grouped["fga"] / grouped["fga"].sum()
    basic = grouped.index.get_level_values("zone_basic")
    grouped["value"] = [zone_value(z) for z in basic]
    grouped["pps"] = grouped["fg_pct"] * grouped["value"]
    return grouped


def zone_table(
    shots: pd.DataFrame,
    league: pd.DataFrame,
    keys: Sequence[str] = BASIC,
    prior_attempts: float = PRIOR_ATTEMPTS,
) -> pd.DataFrame:
    """One row per league zone: the player's volume and efficiency against the league's.

    - ``ci_low``/``ci_high``: 95% Wilson interval of the player's field-goal rate.
    - ``excludes_league``: True when the league rate lies outside that interval (the only case
      where the gap is distinguishable from sampling noise for that zone).
    - ``shrunk_fg_pct``: (fgm + prior_attempts * league_rate) / (fga + prior_attempts): pulls
      small samples towards the league rate; with many shots it converges to the raw rate.
    - ``pps*``: points per shot (field-goal rate x 2 or 3).
    """
    lg = league_zone_table(league, keys)
    mine = _keep(shots).groupby(list(keys)).agg(fga=("made", "size"), fgm=("made", "sum"))
    t = lg.add_prefix("league_").join(mine, how="left")
    t[["fga", "fgm"]] = t[["fga", "fgm"]].fillna(0).astype(int)
    t["value"] = t["league_value"]
    t["fg_pct"] = (t["fgm"] / t["fga"]).where(t["fga"] > 0)
    t["ci_low"], t["ci_high"] = wilson_interval(t["fgm"], t["fga"])
    t["freq"] = t["fga"] / max(t["fga"].sum(), 1)
    t["excludes_league"] = (t["league_fg_pct"] < t["ci_low"]) | (t["league_fg_pct"] > t["ci_high"])
    t["shrunk_fg_pct"] = (t["fgm"] + prior_attempts * t["league_fg_pct"]) / (
        t["fga"] + prior_attempts
    )
    t["pps"] = t["fg_pct"] * t["value"]
    t["pps_low"] = t["ci_low"] * t["value"]
    t["pps_high"] = t["ci_high"] * t["value"]
    t["shrunk_pps"] = t["shrunk_fg_pct"] * t["value"]
    t["pps_diff"] = t["pps"] - t["league_pps"]
    columns = [
        "fga", "fgm", "fg_pct", "ci_low", "ci_high", "shrunk_fg_pct", "excludes_league",
        "freq", "league_freq", "value", "pps", "pps_low", "pps_high", "shrunk_pps",
        "league_fg_pct", "league_pps", "pps_diff",
    ]  # fmt: skip
    return t.rename(columns={"league_share": "league_freq"})[columns]


def selection_vs_making(
    shots: pd.DataFrame, league: pd.DataFrame, keys: Sequence[str] = BASIC
) -> dict[str, float]:
    """Split a shooter's points per shot into WHERE he shoots and HOW WELL he converts.

    - ``league_pps``: league-wide points per shot.
    - ``selection``: points per shot his shot mix would earn at league-average efficiency in each
      zone, minus ``league_pps`` (positive = takes better-than-average shots).
    - ``making``: actual minus that expectation, with a 95% interval from per-shot residuals
      (positive = converts better than league average from the same zones).
    """
    lg = league_zone_table(league, keys)
    league_pps = float((lg["fgm"] * lg["value"]).sum() / lg["fga"].sum())
    s = _keep(shots)
    expected = s.join(lg["pps"].rename("expected"), on=list(keys))["expected"]
    residual = s["made"] * s["points"] - expected
    n = len(s)
    making = float(residual.mean())
    half = 1.96 * float(residual.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "league_pps": league_pps,
        "selection": float(expected.mean()) - league_pps,
        "making": making,
        "making_low": making - half,
        "making_high": making + half,
        "actual_pps": float((s["made"] * s["points"]).mean()),
    }


def half_season_counts(shots: pd.DataFrame, keys: Sequence[str] = BASIC) -> pd.DataFrame:
    """Attempts and makes per zone in the first and the second half of a player's game dates.

    The split is chronological (never shuffled): the first half is what we would know at
    mid-season, the second half is the future we try to predict.
    """
    s = _keep(shots)
    dates = np.sort(s["game_date"].unique())
    cut = dates[len(dates) // 2]
    first = s[s["game_date"] < cut].groupby(list(keys))["made"].agg(fga_1="size", fgm_1="sum")
    second = s[s["game_date"] >= cut].groupby(list(keys))["made"].agg(fga_2="size", fgm_2="sum")
    return first.join(second, how="outer").fillna(0).astype(int)


def shrinkage_error(counts: pd.DataFrame, prior_attempts: float) -> float:
    """Attempt-weighted mean squared error of predicting second-half rates from first-half ones.

    ``counts`` stacks ``half_season_counts`` of many players and has a ``league_rate`` column.
    The prediction is the shrunk first-half rate; ``prior_attempts=0`` is the raw rate. Rows
    without first- or second-half attempts are skipped, so every ``prior_attempts`` is scored on
    the same rows. The second-half rate is itself noisy, but that noise does not depend on the
    predictor, so a lower error still means a better predictor.
    """
    rows = counts[(counts["fga_1"] > 0) & (counts["fga_2"] > 0)]
    predicted = (rows["fgm_1"] + prior_attempts * rows["league_rate"]) / (
        rows["fga_1"] + prior_attempts
    )
    squared = (rows["fgm_2"] / rows["fga_2"] - predicted) ** 2
    return float((rows["fga_2"] * squared).sum() / rows["fga_2"].sum())
