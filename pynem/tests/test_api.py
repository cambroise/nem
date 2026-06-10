"""Tests for the scikit-learn-style API (get_params/set_params/predict/...)."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynem import NEM


def _blobs(n=80, d=3, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack([rng.normal(0, 1, (half, d)), rng.normal(5, 1, (n - half, d))])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    return X, G


def test_get_params_roundtrip():
    est = NEM(n_clusters=3, beta=0.7, family="bernoulli", dispersion="sk_")
    params = est.get_params()
    # enum params come back as their string form
    assert params["family"] == "bernoulli"
    assert params["dispersion"] == "sk_"
    assert params["n_clusters"] == 3 and params["beta"] == 0.7
    # NEM(**params) reconstructs an equivalent estimator
    clone = NEM(**params)
    assert clone.get_params() == params


def test_set_params():
    est = NEM(n_clusters=2)
    est.set_params(n_clusters=4, family="laplace")
    assert est.n_clusters == 4
    assert est.get_params()["family"] == "laplace"
    with pytest.raises(ValueError):
        est.set_params(not_a_param=1)


def test_set_params_enum_takes_effect_through_fit():
    # the enum conversion in set_params must actually change the fitted model
    rng = np.random.default_rng(0)
    n, d = 60, 4
    X = (rng.random((n, d)) < np.where(np.arange(n)[:, None] < 30, 0.9, 0.1)).astype(float)
    G = nx.cycle_graph(n)
    est = NEM(n_clusters=2, beta=0.0, family="normal", dispersion="skd",
              random_state=0)
    est.set_params(family="bernoulli")
    est.fit(X, graph=G)
    # Bernoulli centres are binary modes (0/1) — proves bernoulli was used
    assert set(np.unique(est.centers_)) <= {0.0, 1.0}


def test_clone_like_reproducibility():
    X, G = _blobs()
    est = NEM(n_clusters=2, beta=0.5, family="normal", random_state=0)
    labels_a = est.fit(X, graph=G).labels_
    clone = NEM(**est.get_params())          # fresh estimator, same params
    labels_b = clone.fit(X, graph=G).labels_
    assert np.array_equal(labels_a, labels_b)


def test_predict_and_transform_on_fitted_and_new():
    X, G = _blobs()
    est = NEM(n_clusters=2, beta=0.5, family="normal", random_state=0).fit(X, graph=G)

    # no argument -> fitted-data results
    assert np.array_equal(est.predict(), est.labels_)
    assert np.array_equal(est.transform(), est.membership_)

    # new data -> classification by the fitted mixture
    rng = np.random.default_rng(1)
    Xnew = np.vstack([rng.normal(0, 1, (5, 3)), rng.normal(5, 1, (5, 3))])
    lab = est.predict(Xnew)
    mem = est.transform(Xnew)
    assert lab.shape == (10,)
    assert set(np.unique(lab)) <= {1, 2}
    assert mem.shape == (10, 2)
    assert np.allclose(mem.sum(axis=1), 1.0)
    # the two well-separated groups get opposite labels
    assert lab[:5].tolist().count(lab[0]) == 5
    assert lab[0] != lab[-1]


def test_fit_predict():
    X, G = _blobs()
    est = NEM(n_clusters=2, beta=0.5, random_state=0)
    labels = est.fit_predict(X, graph=G)
    assert np.array_equal(labels, est.labels_)


def test_score():
    X, G = _blobs()
    est = NEM(n_clusters=2, beta=0.5, random_state=0).fit(X, graph=G)
    s = est.score()
    assert isinstance(s, float)
    assert s == pytest.approx(est.criteria_["L"])
    # score on new data is a finite float
    assert np.isfinite(est.score(X[:10]))


def test_methods_require_fit():
    est = NEM(n_clusters=2)
    with pytest.raises(RuntimeError):
        est.predict()
    with pytest.raises(RuntimeError):
        est.transform(np.zeros((3, 2)))
    with pytest.raises(RuntimeError):
        est.score()
