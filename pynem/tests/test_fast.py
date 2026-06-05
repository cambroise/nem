"""The Numba sequential E-step must match the pure-Python fallback exactly."""

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pynem.core as core
from pynem.core import NEM
from pynem._fast import HAS_NUMBA


def _toy():
    """Three Gaussian blobs on a ring graph (continuous -> non-degenerate).

    The kernel only sees ``log_pkfki``, so the family is irrelevant to the
    Numba-vs-Python equivalence; Normal data keeps every iteration finite,
    exercising both the soft (nem) and hardening (ncem) branches cleanly.
    """
    rng = np.random.default_rng(0)
    per, d = 70, 4
    centers = np.array([[0, 0, 0, 0], [6, 6, 6, 6], [12, 0, 12, 0]], float)
    labels = np.repeat([0, 1, 2], per)
    X = centers[labels] + rng.normal(0, 1.0, (len(labels), d))
    n = len(labels)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    order = rng.permutation(n)
    for a, b in zip(order, np.roll(order, 1)):
        G.add_edge(int(a), int(b), weight=round(float(rng.uniform(0.3, 1.0)), 3))
    for _ in range(2 * n):
        a, b = map(int, rng.integers(0, n, 2))
        if a != b:
            G.add_edge(a, b, weight=round(float(rng.uniform(0.3, 1.0)), 3))
    return X, G


def _fit(X, G, algorithm):
    return NEM(n_clusters=3, beta=1.5, family="normal", dispersion="sk_",
               proportion="pk", algorithm=algorithm, init="sort",
               site_update="seq", max_iter=30).fit(X, graph=G)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("algorithm", ["nem", "ncem"])
def test_numba_matches_python_fallback(algorithm):
    """Numba seq E-step must match the pure-Python loop (soft and hardening)."""
    X, G = _toy()
    saved = core.HAS_NUMBA
    try:
        core.HAS_NUMBA = True
        m_nb = _fit(X, G, algorithm)
        core.HAS_NUMBA = False
        m_py = _fit(X, G, algorithm)
    finally:
        core.HAS_NUMBA = saved
    # same float64 ops and visit order; only exp() may differ by ~1 ULP
    assert np.max(np.abs(m_nb.membership_ - m_py.membership_)) < 1e-12
    np.testing.assert_array_equal(m_nb.labels_, m_py.labels_)


def test_python_fallback_runs_without_numba():
    """Forcing numba off must still produce a valid fit (fallback path)."""
    X, G = _toy()
    saved = core.HAS_NUMBA
    try:
        core.HAS_NUMBA = False
        m = _fit(X, G, "nem")
    finally:
        core.HAS_NUMBA = saved
    assert m.membership_.shape == (X.shape[0], 3)
    np.testing.assert_allclose(m.membership_.sum(axis=1), 1.0, atol=1e-10)
