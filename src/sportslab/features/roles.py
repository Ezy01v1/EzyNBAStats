"""Feature matrix describing HOW a player plays (not how good he is), for role clustering."""

from __future__ import annotations

import pandas as pd

PER100_FEATURES = [
    "fga100",
    "fg3a100",
    "fta100",
    "oreb100",
    "dreb100",
    "ast100",
    "tov100",
    "stl100",
    "blk100",
    "pf100",
]
SHARE_FEATURES = ["share_ra", "share_paint", "share_mid", "share_corner3"]  # above-break 3 = rest
ADVANCED_FEATURES = ["usg_pct", "ts_pct"]
FEATURES = [*PER100_FEATURES, *SHARE_FEATURES, *ADVANCED_FEATURES]
# Alternative feature sets: groups that exist for only one of them are an artifact of the choice.
FEATURE_SUBSETS = {
    "solo por 100 posesiones": PER100_FEATURES,
    "solo zonas + uso + TS%": [*SHARE_FEATURES, *ADVANCED_FEATURES],
    "sin zonas de tiro": [*PER100_FEATURES, *ADVANCED_FEATURES],
    "sin faltas ni pérdidas": [f for f in FEATURES if f not in ("pf100", "tov100")],
}


def build_role_features(
    advanced: pd.DataFrame, per100: pd.DataFrame, zones: pd.DataFrame, min_minutes: float = 1000
) -> pd.DataFrame:
    """One row per rotation player: identifiers plus the feature columns.

    ``advanced`` comes from ``load_player_stats`` (minutes, usage, TS%), ``per100`` and ``zones``
    from ``sportslab.data.profiles``. Zone columns become shares of the player's attempts (the
    four we keep; above-the-break threes are the remainder).
    """
    pool = advanced[advanced["minutes"] >= min_minutes]
    df = pool.merge(per100, on="player_id", validate="one_to_one").merge(
        zones, on="player_id", validate="one_to_one"
    )
    total = df[["ra_fga", "paint_fga", "mid_fga", "corner3_fga", "atb3_fga"]].sum(axis=1)
    df = df[total > 0].copy()
    total = total[df.index]
    df["share_ra"] = df["ra_fga"] / total
    df["share_paint"] = df["paint_fga"] / total
    df["share_mid"] = df["mid_fga"] / total
    df["share_corner3"] = df["corner3_fga"] / total
    keep = ["season", "player_id", "player", "team", "minutes", *FEATURES]
    return df[keep].reset_index(drop=True)


def standardize(
    features: pd.DataFrame, reference: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-scores of the ``FEATURES`` columns using mean/std of ``reference`` (default: itself).

    Without this, a variable measured in larger units (e.g. 20 shots per 100) would dominate the
    distances over one measured in fractions (e.g. 0.57 TS%).
    """
    ref = features if reference is None else reference
    mean, std = ref[FEATURES].mean(), ref[FEATURES].std(ddof=0)
    z = (features[FEATURES] - mean) / std
    return z, mean, std
