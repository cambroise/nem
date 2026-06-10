"""Robustness to degenerate classes.

Two distinct failure modes used to make ``NEM.fit`` return ``None`` (then crash
in ``_store_result``):

1. a class **empties** (total membership -> 0), e.g. with hard ICM (ncem) and
   strong spatial smoothing — handled by the k-means++ reinitialisation;
2. a non-empty class has **zero dispersion** on some variable (all its members
   agree, e.g. a gene family present in every genome -> mode 1, dispersion 0),
   giving -inf density to non-members and ``0·(-inf) = NaN`` in the D criterion
   — handled by masking ``c_ik > 0`` in D.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynem.core import NEM
from pynem.models import (
    EMPTY_CLASS_WEIGHT,
    Dispersion,
    Family,
    Proportion,
    estimate_parameters,
)


def test_reinit_empty_class_unit():
    """An empty class is moved to the point farthest from the dominant centre."""
    # Two well-separated groups, K=3 with class index 2 left empty.
    X = np.array([[0.0, 0.0], [0.2, 0.1],
                  [10.0, 10.0], [10.2, 9.9]])
    C = np.array([[1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0]], dtype=float)
    p = estimate_parameters(X, C, Family.NORMAL, Dispersion.SK_, Proportion.FREE)

    # class 2 was empty -> revived: real centre, finite dispersion, non-zero prop
    assert (C.sum(axis=0)[2]) < EMPTY_CLASS_WEIGHT
    assert p["proportions"][2] > 0
    assert np.all(np.isfinite(p["dispersions"][2]))
    assert np.all(p["proportions"] > 0)
    # the revived centre is an actual observation (farthest from the dominant)
    assert np.any(np.all(p["centers"][2] == X, axis=1))


def test_degenerate_ncem_does_not_crash():
    """ncem + strong beta on binary data empties a class; fit must recover."""
    rng = np.random.default_rng(0)
    n, d = 200, 12
    X = (rng.random((n, d)) < rng.random((n, 1))).astype(float)
    G = nx.random_regular_graph(4, n, seed=1)
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0

    m = NEM(n_clusters=3, beta=1.5, family="normal", dispersion="sk_",
            proportion="pk", algorithm="ncem", init="sort", site_update="seq",
            max_iter=50).fit(X, graph=G)

    assert m.labels_ is not None
    assert np.isfinite(m.criteria_["U"])
    assert np.isfinite(m.criteria_["D"])
    assert m.n_iter_ <= 50                       # terminates (no infinite oscillation)
    np.testing.assert_allclose(m.membership_.sum(axis=1), 1.0, atol=1e-10)


def test_zero_dispersion_with_members_finite_criteria():
    """A class with zero dispersion (a constant feature) must not NaN the D crit."""
    # variable 0 is constant within each block -> mode {1,0}, dispersion 0,
    # so non-members get -inf density under the other class.
    XA = np.ones((6, 3))
    XB = np.zeros((6, 3))
    X = np.vstack([XA, XB]).astype(float)
    G = nx.path_graph(12)
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0

    m = NEM(n_clusters=2, beta=0.5, family="bernoulli", dispersion="sk_",
            proportion="pk", algorithm="ncem", init="sort", site_update="seq",
            missing="ignore", max_iter=30).fit(X, graph=G)

    assert np.isfinite(m.criteria_["D"])
    assert np.isfinite(m.criteria_["U"])
    assert set(m.labels_) == {1, 2}              # both blocks recovered
