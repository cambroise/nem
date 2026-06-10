"""Tests for the weighted NEM (per-variable feature weights w_j)."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynem import NEM
from pynem.metrics import adjusted_rand_index as ari
from pynem.models import Dispersion, Family, Proportion, compute_log_density, estimate_parameters


def _ring_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=1.0)
    return G


def _gaussian_blobs(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack([rng.normal(0, 1, (half, d)),
                   rng.normal(4, 1, (n - half, d))])
    return X, _ring_graph(n)


def _bernoulli_blocks(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    lab = np.array([0] * half + [1] * (n - half))
    p = np.where(lab[:, None] == 0, 0.85, 0.15)
    X = (rng.random((n, d)) < p).astype(float)
    return X, _ring_graph(n), lab


# --- 1. w = ones (explicit) reduces EXACTLY to the unweighted default --------

@pytest.mark.parametrize("family,disp,data", [
    ("normal", "s__", "gauss"),
    ("normal", "sk_", "gauss"),
    ("normal", "skd", "gauss"),
    ("bernoulli", "sk_", "bern"),
    ("bernoulli", "skd", "bern"),
])
def test_ones_equals_unweighted(family, disp, data):
    if data == "gauss":
        X, G = _gaussian_blobs()
    else:
        X, G, _ = _bernoulli_blocks()
    d = X.shape[1]

    common = dict(n_clusters=2, beta=0.7, family=family, dispersion=disp,
                  proportion="pk", site_update="seq", max_iter=30,
                  random_state=0)
    m0 = NEM(**common).fit(X, graph=G)
    m1 = NEM(feature_weights=np.ones(d), **common).fit(X, graph=G)

    assert np.array_equal(m0.labels_, m1.labels_)
    assert np.max(np.abs(m0.membership_ - m1.membership_)) < 1e-12
    assert np.max(np.abs(m0.centers_ - m1.centers_)) < 1e-12
    assert np.max(np.abs(m0.dispersions_ - m1.dispersions_)) < 1e-12


# --- 2. Weighting changes the result for the pooled dispersion models -------

def test_weights_change_pooled_dispersion():
    X, G = _gaussian_blobs(d=4)
    w = np.array([5.0, 1.0, 1.0, 1.0])  # over-weight the first variable
    common = dict(n_clusters=2, beta=0.5, family="normal", dispersion="sk_",
                  proportion="pk", site_update="seq", max_iter=30,
                  random_state=0)
    m0 = NEM(**common).fit(X, graph=G)
    mw = NEM(feature_weights=w, **common).fit(X, graph=G)
    # sk_ pools across variables -> dispersions must differ with non-uniform w
    assert np.max(np.abs(m0.dispersions_ - mw.dispersions_)) > 1e-6


# --- 3. The note's scenario: down-weighting redundant columns recovers truth -

def test_redundant_columns_corrected_by_weights():
    rng = np.random.default_rng(0)
    n = 400
    t = rng.integers(0, 2, n)        # true signal, carried by 6 columns
    s = rng.integers(0, 2, n)        # spurious signal, carried by 60 columns

    def block(label, ncol, p1=0.9, p0=0.1):
        P = np.where(label[:, None] == 1, p1, p0)
        return (rng.random((n, ncol)) < P).astype(float)

    X = np.hstack([block(t, 6), block(s, 60)])
    G = _ring_graph(n)
    w = np.concatenate([np.ones(6), np.full(60, 1 / 60)])  # inverse-abundance

    common = dict(n_clusters=2, beta=0.0, family="bernoulli", dispersion="skd",
                  proportion="pk", init="random", n_init=4, site_update="seq",
                  random_state=1, max_iter=100)
    lab_u = NEM(**common).fit(X, graph=G).labels_
    lab_w = NEM(feature_weights=w, **common).fit(X, graph=G).labels_

    # unweighted locks onto the redundant (spurious) structure
    assert ari(s, lab_u) > 0.8
    assert ari(t, lab_u) < 0.3
    # weighted recovers the true structure
    assert ari(t, lab_w) > 0.8


# --- 4. w_j = 0 ignores a variable (selection as a limiting case) ------------

def test_zero_weight_ignores_variable():
    # var 0 carries the truth; var 1 is noise. Centres differ only in var 0,
    # so weighting it out (w0=0) must make both classes equally likely.
    X = np.array([[0.0, 0.3], [5.0, -0.2], [0.1, 1.0], [4.9, 0.5]])
    ld = compute_log_density(
        X, np.array([[0.0, 0.0], [5.0, 0.0]]), np.ones((2, 2)),
        np.array([0.5, 0.5]), Family.NORMAL, weights=np.array([0.0, 1.0]))
    assert np.max(np.abs(ld[:, 0] - ld[:, 1])) < 1e-9


def test_zero_weight_ignores_degenerate_column():
    # The case the pangenome cares about: a column with ZERO dispersion
    # (point mass). Without the weight-aware mask an off-mode point would get
    # -inf density; w_j=0 must instead ignore the column entirely.
    X = np.array([[1.0, 0.0],     # off the class-1 mode in the degenerate col 0
                  [0.0, 1.0]])
    centers = np.array([[0.0, 0.0], [0.0, 1.0]])
    disp = np.array([[0.0, 0.5], [0.0, 0.5]])   # col 0 dispersion = 0 (degenerate)
    prop = np.array([0.5, 0.5])

    # active weight on the degenerate col -> off-mode point is invalid (-inf)
    ld_on = compute_log_density(X, centers, disp, prop, Family.BERNOULLI,
                                weights=np.array([1.0, 1.0]))
    assert np.isneginf(ld_on[0, 0])

    # weight it out -> finite density, column ignored
    ld_off = compute_log_density(X, centers, disp, prop, Family.BERNOULLI,
                                 weights=np.array([0.0, 1.0]))
    assert np.all(np.isfinite(ld_off))


def test_partition_pangenome_unit_weights_match_default():
    rng = np.random.default_rng(0)
    n, n_org = 120, 12
    lab = np.array([0] * 40 + [1] * 40 + [2] * 40)
    p = np.choose(lab, [0.95, 0.5, 0.08])
    X = (rng.random((n, n_org)) < p[:, None]).astype(float)
    G = _ring_graph(n)

    from pynem import partition_pangenome
    r0 = partition_pangenome(X, G, K=3, beta=2.5, max_iter=50)
    r1 = partition_pangenome(X, G, K=3, beta=2.5, max_iter=50,
                             genome_weights=np.ones(n_org))
    assert np.array_equal(r0["partition"], r1["partition"])
    assert np.max(np.abs(r0["membership"] - r1["membership"])) < 1e-12

    # a non-uniform weighting must be able to change the partition
    w = np.concatenate([np.full(n_org - 3, 0.2), np.full(3, 3.0)])
    rw = partition_pangenome(X, G, K=3, beta=2.5, max_iter=50,
                             genome_weights=w)
    assert rw["membership"].shape == r0["membership"].shape


# --- 5. validation -----------------------------------------------------------

def test_weight_shape_validation():
    X, G = _gaussian_blobs(d=4)
    with pytest.raises(ValueError):
        NEM(n_clusters=2, feature_weights=np.ones(3)).fit(X, graph=G)
    with pytest.raises(ValueError):
        NEM(n_clusters=2, feature_weights=np.array([1, 1, -1, 1.0])).fit(
            X, graph=G)


# --- 6. M-step center invariance under per-column weighting ------------------

def test_centers_invariant_under_weighting():
    X, _, _ = _bernoulli_blocks(d=5)
    C = np.zeros((X.shape[0], 2))
    C[:30, 0] = 1.0
    C[30:, 1] = 1.0
    w = np.array([3.0, 0.2, 1.0, 5.0, 0.5])
    p0 = estimate_parameters(X, C, Family.BERNOULLI, Dispersion.SKD,
                             Proportion.FREE, miss_mode="ignore")
    pw = estimate_parameters(X, C, Family.BERNOULLI, Dispersion.SKD,
                             Proportion.FREE, miss_mode="ignore", weights=w)
    # Bernoulli modes (centres) and skd dispersions are unchanged by w_j
    assert np.array_equal(p0["centers"], pw["centers"])
    assert np.max(np.abs(p0["dispersions"] - pw["dispersions"])) < 1e-12
