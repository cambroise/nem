"""Tests for the MAG-aware (completeness) Bernoulli emission."""

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynem import NEM, partition_pangenome
from pynem.models import compute_log_density, estimate_parameters, Family, \
    Dispersion, Proportion


def _ring(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=1.0)
    return G


def _pangenome(n=150, d=14, seed=0):
    """Presence matrix with a clear persistent / shell / cloud structure."""
    rng = np.random.default_rng(seed)
    third = n // 3
    lab = np.array([0] * third + [1] * third + [2] * (n - 2 * third))
    p = np.choose(lab, [0.97, 0.5, 0.06])
    X = (rng.random((n, d)) < p[:, None]).astype(float)
    return X, _ring(n), lab


# --- 1. completeness=None is the exact default ------------------------------

def test_completeness_none_is_default():
    X, G, _ = _pangenome()
    r0 = partition_pangenome(X, G, K=3, beta=2.5, max_iter=40)
    r1 = partition_pangenome(X, G, K=3, beta=2.5, max_iter=40, completeness=None)
    assert np.array_equal(r0["partition"], r1["partition"])
    assert np.max(np.abs(r0["membership"] - r1["membership"])) == 0.0


# --- 2. completeness restores the persistent class on eroded MAGs -----------

def test_completeness_restores_persistent():
    X, G, _ = _pangenome(n=300, d=16, seed=1)
    ref = partition_pangenome(X, G, K=3, beta=2.5, max_iter=60)["partition"]
    ref_P = ref == "P"
    assert ref_P.sum() > 20                      # a real persistent class exists

    # erode every genome to known completeness (independent dropout)
    rng = np.random.default_rng(2)
    gamma = rng.uniform(0.4, 0.7, X.shape[1])
    Xe = X.copy()
    for j in range(X.shape[1]):
        drop = (rng.random(X.shape[0]) < (1 - gamma[j])) & (Xe[:, j] == 1)
        Xe[drop, j] = 0.0

    naive = partition_pangenome(Xe, G, K=3, beta=2.5, max_iter=60)["partition"]
    comp = partition_pangenome(Xe, G, K=3, beta=2.5, max_iter=60,
                               completeness=gamma)["partition"]
    n_naive = int((naive == "P").sum())
    n_comp = int((comp == "P").sum())
    # naive collapses the persistent class; completeness recovers most of it
    assert n_comp > n_naive
    recovered = int(((comp == "P") & ref_P).sum())
    assert recovered >= 0.8 * ref_P.sum()


# --- 3. emission: an absence in an incomplete genome is forgiven -------------

def test_emission_forgives_absence_in_incomplete_genome():
    # one persistent gene (mode 1 everywhere), observed absent in genome 0
    X = np.array([[0.0, 1.0, 1.0]])              # absent in genome 0
    centers = np.ones((1, 3))                    # mode 1 (present) everywhere
    disp = np.full((1, 3), 0.05)
    prop = np.array([1.0])
    full = compute_log_density(X, centers, disp, prop, Family.BERNOULLI,
                               completeness=np.array([1.0, 1.0, 1.0]))
    incomplete = compute_log_density(X, centers, disp, prop, Family.BERNOULLI,
                                     completeness=np.array([0.3, 1.0, 1.0]))
    # the absence in genome 0 costs less when genome 0 is known incomplete
    assert incomplete[0, 0] > full[0, 0]


# --- 4. M-step: mode survives erosion with completeness ---------------------

def test_mstep_mode_preserved_under_completeness():
    # gene present in only 40% of a class's genomes (eroded), gamma=0.4
    n = 100
    C = np.ones((n, 1))
    X = np.zeros((n, 1))
    X[:40] = 1.0                                  # 40% present
    p_naive = estimate_parameters(X, C, Family.BERNOULLI, Dispersion.SKD,
                                  Proportion.FREE, miss_mode="ignore")
    p_comp = estimate_parameters(X, C, Family.BERNOULLI, Dispersion.SKD,
                                 Proportion.FREE, miss_mode="ignore",
                                 completeness=np.array([0.4]))
    # naive: 40% < 0.5 -> mode 0; completeness: 0.4/0.4 = 1 > 0.5 -> mode 1
    assert p_naive["centers"][0, 0] == 0.0
    assert p_comp["centers"][0, 0] == 1.0


# --- 5. self-estimated completeness ("auto", no CheckM) ---------------------

def test_completeness_auto_recovers_persistent():
    X, G, _ = _pangenome(n=300, d=16, seed=3)
    ref_P = partition_pangenome(X, G, K=3, beta=2.5, max_iter=60)["partition"] == "P"
    rng = np.random.default_rng(4)
    gamma = rng.uniform(0.4, 0.7, X.shape[1])
    Xe = X.copy()
    for j in range(X.shape[1]):
        drop = (rng.random(X.shape[0]) < (1 - gamma[j])) & (Xe[:, j] == 1)
        Xe[drop, j] = 0.0

    naive = (partition_pangenome(Xe, G, K=3, beta=2.5, max_iter=60)["partition"] == "P").sum()
    res = partition_pangenome(Xe, G, K=3, beta=2.5, max_iter=60, completeness="auto")
    auto_P = res["partition"] == "P"
    # self-estimation returns the estimated completeness and recovers persistent
    assert res["completeness"].shape == (X.shape[1],)
    assert res["completeness_n_iter"] >= 1
    assert int(auto_P.sum()) > int(naive)
    assert int((auto_P & ref_P).sum()) >= 0.8 * ref_P.sum()
    # estimated completeness correlates with the (hidden) true completeness
    assert np.corrcoef(res["completeness"], gamma)[0, 1] > 0.7


def test_completeness_bad_string_raises():
    X, G, _ = _pangenome()
    with pytest.raises(ValueError):
        partition_pangenome(X, G, K=3, completeness="nope")


# --- 6. validation ----------------------------------------------------------

def test_completeness_validation():
    X, G, _ = _pangenome()
    d = X.shape[1]
    with pytest.raises(ValueError):              # wrong shape
        NEM(n_clusters=3, family="bernoulli",
            completeness=np.ones(d - 1)).fit(X, graph=G)
    with pytest.raises(ValueError):              # out of (0, 1]
        NEM(n_clusters=3, family="bernoulli",
            completeness=np.full(d, 1.5)).fit(X, graph=G)
    with pytest.raises(ValueError):              # not Bernoulli
        NEM(n_clusters=3, family="normal",
            completeness=np.ones(d)).fit(X, graph=G)
