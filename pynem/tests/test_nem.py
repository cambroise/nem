"""Tests for pynem, using the essai example data."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path for editable install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pynem
from pynem import NEM
from pynem.models import Family, Dispersion, Proportion

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
BASENAME = str(EXAMPLES_DIR / "essai")


class TestIO:
    def test_read_str(self):
        t, n, d = pynem.io.read_str(BASENAME + ".str")
        assert t == "S"
        assert n == 100
        assert d == 3

    def test_read_str_image(self):
        basename = str(EXAMPLES_DIR / "potts_30x30_3")
        t, n, d = pynem.io.read_str(basename + ".str")
        assert t == "I"
        assert n == 900  # 30 * 30
        assert d == 2

    def test_read_dat(self):
        _, n, d = pynem.io.read_str(BASENAME + ".str")
        X = pynem.io.read_dat(BASENAME + ".dat", n, d)
        assert X.shape == (100, 3)
        assert not np.isnan(X).any()

    def test_read_nei(self):
        _, n, _ = pynem.io.read_str(BASENAME + ".str")
        G = pynem.io.read_nei(BASENAME + ".nei", n)
        assert G.number_of_nodes() == 100
        assert G.number_of_edges() > 0

    def test_read_graph(self):
        G = pynem.io.read_graph(BASENAME)
        assert G.number_of_nodes() == 100
        assert G.graph["d"] == 3
        # Check features are set
        feat = G.nodes[0]["features"]
        assert len(feat) == 3
        assert isinstance(feat, np.ndarray)
        # Check connectivity
        assert G.number_of_edges() > 50


class TestPureEM:
    """NEM with beta=0 reduces to standard EM."""

    def test_pure_em_converges(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="normal",
                    init="sort", max_iter=100, verbose=0)
        model.fit(G)
        assert model.labels_ is not None
        assert len(model.labels_) == 100
        assert model.n_iter_ > 1
        assert model.n_iter_ <= 100

    def test_pure_em_criteria(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="normal", init="sort")
        model.fit(G)
        # With beta=0, U = D and G should still be computed
        assert model.criteria_["U"] == pytest.approx(model.criteria_["D"])
        assert np.isfinite(model.criteria_["L"])


class TestNEMSpatial:
    def test_nem_with_beta(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.1, family="normal", init="sort")
        model.fit(G)
        assert len(model.labels_) == 100
        assert set(model.labels_).issubset({1, 2, 3})
        # U should be different from D when beta > 0
        assert model.criteria_["U"] != pytest.approx(model.criteria_["D"])
        assert model.criteria_["G"] > 0


class TestNCEM:
    def test_ncem(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.1, algorithm="ncem",
                    family="normal", init="sort")
        model.fit(G)
        # NCEM produces hard labels; membership should be 0/1
        assert np.all((model.membership_ == 0) | (model.membership_ == 1))
        assert np.all(model.membership_.sum(axis=1) == 1)


class TestFuzzyOutput:
    def test_membership_sums_to_one(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.1, family="normal", init="sort")
        model.fit(G)
        row_sums = model.membership_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_membership_shape(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="normal", init="sort")
        model.fit(G)
        assert model.membership_.shape == (100, 3)


class TestSortInit:
    def test_sort_init_balanced(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="normal", init="sort",
                    max_iter=1)
        model.fit(G)
        # After sort init + 1 iteration, should have all 3 classes
        unique_labels = set(model.labels_)
        assert len(unique_labels) == 3


class TestModelVariants:
    @pytest.mark.parametrize("disp", ["s__", "sk_", "s_d", "skd"])
    def test_dispersion_models(self, disp):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="normal",
                    dispersion=disp, init="sort", max_iter=20)
        model.fit(G)
        assert model.dispersions_.shape == (3, 3)
        assert np.all(model.dispersions_ > 0)

    @pytest.mark.parametrize("prop", ["p_", "pk"])
    def test_proportion_models(self, prop):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="normal",
                    proportion=prop, init="sort", max_iter=20)
        model.fit(G)
        np.testing.assert_allclose(model.proportions_.sum(), 1.0, atol=1e-10)
        if prop == "p_":
            np.testing.assert_allclose(model.proportions_,
                                       np.full(3, 1.0 / 3), atol=1e-10)

    def test_laplace_family(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="laplace",
                    init="sort", max_iter=20)
        model.fit(G)
        assert len(model.labels_) == 100


class TestHeuristicBeta:
    def test_heuristic_d(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta_mode="heu_d", family="normal",
                    init="sort", max_iter=50)
        model.fit(G)
        assert model.beta_ >= 0
        assert len(model.labels_) == 100


class TestRandomInit:
    def test_random_init(self):
        G = pynem.io.read_graph(BASENAME)
        model = NEM(n_clusters=3, beta=0.0, family="normal",
                    init="random", n_init=3, random_state=42, max_iter=30)
        model.fit(G)
        assert len(model.labels_) == 100
        assert set(model.labels_).issubset({1, 2, 3})
