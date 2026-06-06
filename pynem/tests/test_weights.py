"""Tests for automatic weight derivation (Jaccard + UPGMA -> abundance weights)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynem.weights import (
    jaccard_upgma_linkage, silhouette_score, jaccard_upgma_labels,
    abundance_weights, genome_weights,
)


def _clustered_presence(seed=0):
    """A presence matrix with 3 clear genome clades (some over-sampled)."""
    rng = np.random.default_rng(seed)
    n_fam = 200
    # three latent genome profiles over the families
    profiles = (rng.random((3, n_fam)) < [[0.8], [0.4], [0.1]]).astype(float)
    # clade sizes 30 / 6 / 2  (clade 0 is over-sampled)
    sizes = [30, 6, 2]
    cols = []
    truth = []
    for g, sz in enumerate(sizes):
        for _ in range(sz):
            # genome = profile + a little per-genome noise
            flip = rng.random(n_fam) < 0.03
            cols.append(np.where(flip, 1 - profiles[g], profiles[g]))
            truth.append(g)
    X = np.column_stack(cols)            # (n_fam, n_genomes)
    return X, np.array(truth)


# --- Jaccard distance / linkage sanity --------------------------------------

def test_jaccard_distance_sanity():
    # two identical genomes -> distance 0; disjoint -> distance 1
    X = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [0, 0, 1]]).T  # rows: g0=g1 identical, g2 disjoint
    cond, Z = jaccard_upgma_linkage(X)
    from scipy.spatial.distance import squareform
    D = squareform(cond)
    assert D[0, 1] == pytest.approx(0.0)
    assert D[0, 2] == pytest.approx(1.0)
    assert Z.shape == (X.shape[0] - 1, 4)


def test_all_zero_pair_no_nan():
    X = np.zeros((3, 5))            # all-zero genomes (0/0 Jaccard)
    cond, Z = jaccard_upgma_linkage(X)
    assert np.all(np.isfinite(cond))


# --- abundance weights invariants -------------------------------------------

def test_abundance_weights_sum_to_d_and_positive():
    labels = np.array([0, 0, 0, 0, 1, 2, 2])   # m=3 groups, d=7
    w = abundance_weights(labels)
    assert w.shape == (7,)
    assert np.all(w > 0)
    assert np.sum(w) == pytest.approx(len(labels))
    # each group totals d/m
    d, m = 7, 3
    for g in np.unique(labels):
        assert w[labels == g].sum() == pytest.approx(d / m)
    # the over-sampled group (4 members) has the smallest per-item weight
    assert w[0] < w[4]


def test_singleton_groups_weight_one():
    labels = np.array([0, 1, 2, 3])            # all singletons -> w_j = d/m = 1
    w = abundance_weights(labels)
    assert np.allclose(w, 1.0)


# --- clustering: identical genomes grouped, fixed vs silhouette -------------

def test_identical_genomes_same_group():
    X, truth = _clustered_presence()
    labels = jaccard_upgma_labels(X.T, n_clusters=3, selection="fixed")
    # the 3 latent clades should be recovered as 3 groups
    assert len(np.unique(labels)) == 3
    # genomes from the same true clade share a label (contiguous blocks)
    for g in np.unique(truth):
        block = labels[truth == g]
        assert len(np.unique(block)) == 1


def test_fixed_n_clusters_respected_and_capped():
    X, _ = _clustered_presence()
    n_genomes = X.shape[1]
    assert len(np.unique(jaccard_upgma_labels(X.T, n_clusters=5))) <= 5
    # asking for more groups than genomes is capped, not an error
    lab = jaccard_upgma_labels(X.T, n_clusters=n_genomes + 50)
    assert len(np.unique(lab)) <= n_genomes


def test_silhouette_selection_returns_valid_grouping():
    X, truth = _clustered_presence()
    labels = jaccard_upgma_labels(X.T, selection="silhouette",
                                  k_range=range(2, 8))
    k = len(np.unique(labels))
    assert 2 <= k <= 7
    # with 3 well-separated clades silhouette should find exactly 3
    assert k == 3


def test_silhouette_score_bounds_and_degenerate():
    from scipy.spatial.distance import pdist
    X, _ = _clustered_presence()
    D = pdist(X.T.astype(bool), metric="jaccard")
    labels = jaccard_upgma_labels(X.T, n_clusters=3)
    s = silhouette_score(D, labels)
    assert -1.0 <= s <= 1.0
    # a single cluster is degenerate -> -1
    assert silhouette_score(D, np.zeros(X.shape[1], dtype=int)) == -1.0


def test_bad_selection_raises():
    X, _ = _clustered_presence()
    with pytest.raises(ValueError):
        jaccard_upgma_labels(X.T, selection="nope")


# --- end-to-end genome_weights ----------------------------------------------

def test_genome_weights_downweights_oversampled_clade():
    X, truth = _clustered_presence()              # clade 0 has 30 genomes
    w, labels = genome_weights(X, n_groups=3, return_labels=True)
    assert w.shape == (X.shape[1],)
    assert np.sum(w) == pytest.approx(X.shape[1])
    assert np.all(w > 0)
    # genomes of the over-sampled clade get a much smaller weight than rares
    big = truth == 0
    small = truth == 2
    assert w[big].mean() < w[small].mean()


def test_genome_weights_silhouette_runs():
    X, _ = _clustered_presence()
    w = genome_weights(X, selection="silhouette", k_range=range(2, 8))
    assert w.shape == (X.shape[1],)
    assert np.sum(w) == pytest.approx(X.shape[1])


def test_genome_weights_feeds_partition_pangenome():
    # end-to-end seam: derived weights plug into partition_pangenome
    import networkx as nx
    from pynem import partition_pangenome
    X, _ = _clustered_presence()
    n_genomes = X.shape[1]
    w = genome_weights(X, n_groups=4)
    assert w.shape == (n_genomes,)
    G = nx.cycle_graph(X.shape[0])
    res = partition_pangenome(X, G, K=3, beta=2.5, max_iter=30,
                              genome_weights=w)
    assert res["partition"].shape == (X.shape[0],)
    assert set(np.unique(res["partition"])) <= {"P", "S", "C"}
