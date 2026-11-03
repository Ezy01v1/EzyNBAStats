"""K-means role clusters and the checks that tell real structure from an artifact.

k-means ALWAYS returns k groups, whether or not the data has groups. So every clustering here is
reported next to (1) silhouette against a null with the same marginals but no joint structure,
(2) bootstrap stability and (3) agreement across seasons.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


def fit_kmeans(z: pd.DataFrame | np.ndarray, k: int, seed: int = 0, n_init: int = 10) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(z)


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters 0, 1, 2... from the largest to the smallest (stable display order)."""
    order = pd.Series(labels).value_counts().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels])


def permuted_reference(z: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle every column independently: same marginals, no relation between variables."""
    return pd.DataFrame(
        {c: rng.permutation(z[c].to_numpy()) for c in z.columns}, index=z.index, columns=z.columns
    )


def bootstrap_stability(z: pd.DataFrame, k: int, n_boot: int = 100, seed: int = 0) -> np.ndarray:
    """Adjusted Rand index between the full-data clustering and ones refit on resampled players.

    1 = same groups every time; ~0 = groups agree no more than chance.
    """
    base = fit_kmeans(z, k, seed).labels_
    rng = np.random.default_rng(seed)
    aris = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(z), size=len(z))
        model = fit_kmeans(z.iloc[idx], k, seed=b, n_init=3)
        aris[b] = adjusted_rand_score(base, model.predict(z))
    return aris


def k_diagnostics(
    z: pd.DataFrame,
    ks: Sequence[int],
    n_boot: int = 100,
    n_null: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Per k: inertia, silhouette, null silhouette, and bootstrap ARI (mean and 5th percentile)."""
    rng = np.random.default_rng(seed)
    nulls = [permuted_reference(z, rng) for _ in range(n_null)]
    rows = []
    for k in ks:
        model = fit_kmeans(z, k, seed)
        null_sil = [silhouette_score(nz, fit_kmeans(nz, k, seed, n_init=3).labels_) for nz in nulls]
        aris = bootstrap_stability(z, k, n_boot, seed)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(z, model.labels_),
                "null_silhouette": float(np.mean(null_sil)),
                "ari_mean": float(aris.mean()),
                "ari_p05": float(np.quantile(aris, 0.05)),
            }
        )
    return pd.DataFrame(rows)


def cross_season_ari(z_old: pd.DataFrame, z_new: pd.DataFrame, k: int, seed: int = 0) -> float:
    """Fit on the old season, assign the new season's players to those centers, and compare with
    clustering the new season directly. Each season is standardized on its own."""
    carried = fit_kmeans(z_old, k, seed).predict(z_new)
    direct = fit_kmeans(z_new, k, seed).labels_
    return float(adjusted_rand_score(direct, carried))


def feature_set_agreement(
    z: pd.DataFrame, subsets: dict[str, Sequence[str]], k: int, seed: int = 0
) -> pd.Series:
    """ARI between the clustering with every feature and the one with each feature subset."""
    base = fit_kmeans(z, k, seed).labels_
    return pd.Series(
        {
            name: adjusted_rand_score(base, fit_kmeans(z[list(cols)], k, seed).labels_)
            for name, cols in subsets.items()
        }
    )


def cluster_profile(z: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean z-score of every feature per cluster, plus the cluster size."""
    profile = z.groupby(labels).mean()
    profile.insert(0, "size", pd.Series(labels).value_counts().sort_index().to_numpy())
    return profile


def principal_components(z: pd.DataFrame, n: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """Coordinates on the first ``n`` principal components, and their explained variance ratio."""
    pca = PCA(n_components=n).fit(z)
    coords = pd.DataFrame(pca.transform(z), index=z.index, columns=[f"PC{i + 1}" for i in range(n)])
    return coords, pca.explained_variance_ratio_


def nearest_players(z: pd.DataFrame, names: pd.Series, player: str, n: int = 5) -> pd.DataFrame:
    """The ``n`` players closest to ``player`` in style (Euclidean distance between z-scores)."""
    row = names.reset_index(drop=True).eq(player)
    if not row.any():
        raise KeyError(player)
    target = z.to_numpy()[row.to_numpy()][0]
    dist = np.linalg.norm(z.to_numpy() - target, axis=1)
    out = pd.DataFrame({"player": names.to_numpy(), "distance": dist})
    return out[out["player"] != player].nsmallest(n, "distance").reset_index(drop=True)
