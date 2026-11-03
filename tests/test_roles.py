import numpy as np
import pandas as pd
import pytest

from sportslab.analysis.roles import (
    bootstrap_stability,
    cluster_profile,
    cross_season_ari,
    feature_set_agreement,
    fit_kmeans,
    k_diagnostics,
    nearest_players,
    permuted_reference,
    principal_components,
    relabel_by_size,
)


def _blobs(seed: int = 0, per_cluster: int = 60, sd: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0, 0], [5, 5, 0, 0], [0, 5, 5, 5]], dtype=float)
    data = np.vstack([c + rng.normal(0, sd, size=(per_cluster, 4)) for c in centers])
    return pd.DataFrame(data, columns=list("abcd"))


def _noise(seed: int = 0, n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))


def test_permuted_reference_keeps_marginals_and_breaks_correlation():
    z = _blobs()
    null = permuted_reference(z, np.random.default_rng(1))
    for c in z.columns:
        assert np.allclose(np.sort(z[c]), np.sort(null[c]))
    assert abs(z.corr().loc["a", "b"]) > 0.4 and abs(null.corr().loc["a", "b"]) < 0.2


def test_clear_clusters_are_stable_and_beat_the_null():
    diag = k_diagnostics(_blobs(), ks=[3], n_boot=20, n_null=5).iloc[0]
    assert diag["silhouette"] > 0.6 and diag["silhouette"] > diag["null_silhouette"] + 0.2
    assert diag["ari_mean"] > 0.95


def test_no_structure_gives_unstable_clusters_and_a_silhouette_close_to_the_null():
    diag = k_diagnostics(_noise(), ks=[3], n_boot=20, n_null=5).iloc[0]
    assert diag["silhouette"] < 0.3
    assert abs(diag["silhouette"] - diag["null_silhouette"]) < 0.1
    assert diag["ari_mean"] < 0.9


def test_bootstrap_stability_is_between_minus_one_and_one():
    aris = bootstrap_stability(_blobs(), k=3, n_boot=10)
    assert len(aris) == 10 and ((aris <= 1) & (aris >= -1)).all()


def test_same_structure_in_two_seasons_gives_high_cross_season_agreement():
    assert cross_season_ari(_blobs(seed=1), _blobs(seed=2), k=3) > 0.95


def test_relabel_by_size_orders_clusters_from_largest():
    labels = np.array([2, 2, 2, 0, 0, 1])
    assert relabel_by_size(labels).tolist() == [0, 0, 0, 1, 1, 2]


def test_cluster_profile_reports_size_and_mean_per_feature():
    z = _blobs()
    labels = fit_kmeans(z, 3).labels_
    prof = cluster_profile(z, labels)
    assert prof["size"].sum() == len(z) and set(prof.columns) == {"size", "a", "b", "c", "d"}
    assert prof["size"].tolist() == [60, 60, 60]


def test_principal_components_report_explained_variance():
    coords, explained = principal_components(_blobs())
    assert coords.shape == (180, 2) and 0 < explained.sum() <= 1 and explained[0] >= explained[1]


def test_nearest_players_excludes_the_player_and_sorts_by_distance():
    z = pd.DataFrame({"a": [0.0, 0.1, 5.0, 0.2], "b": [0.0, 0.0, 5.0, 0.0]})
    names = pd.Series(["A", "B", "C", "D"])
    out = nearest_players(z, names, "A", n=3)
    assert out["player"].tolist() == ["B", "D", "C"]
    with pytest.raises(KeyError):
        nearest_players(z, names, "Z")


def test_feature_subsets_agree_when_structure_is_in_all_features_and_not_when_it_is_in_one():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 5, 0], [5, 5, 0, 5], [0, 5, 5, 5]], dtype=float)
    z = pd.DataFrame(
        np.vstack([c + rng.normal(0, 0.4, size=(60, 4)) for c in centers]), columns=list("abcd")
    )
    agree = feature_set_agreement(z, {"a,b": ["a", "b"], "c,d": ["c", "d"]}, k=3)
    assert (agree > 0.9).all()  # every pair of columns carries the 3 groups
    rng = np.random.default_rng(0)
    only_a = pd.DataFrame(
        {
            "a": np.r_[rng.normal(0, 0.3, 90), rng.normal(6, 0.3, 90)],
            "b": rng.normal(size=180),
            "c": rng.normal(size=180),
        }
    )
    agree = feature_set_agreement(only_a, {"a": ["a"], "b,c": ["b", "c"]}, k=2)
    assert agree["a"] > 0.9 and agree["b,c"] < 0.3
