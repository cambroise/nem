"""Validate that pynem reproduces the PPanGGOLiN NEM partitioning pipeline.

These tests compare pynem against the *reference* PPanGGOLiN implementation
(``ppanggolin.nem.partition.run_partitioning``, which drives the embedded NEM C
core via Cython). They are skipped when PPanGGOLiN is not installed.

The comparison is **deterministic** (parameter-file init, no RNG on either
side) and **element-wise** on the membership matrix and the parameters — not on
argmax labels, which agree at the fixed point regardless of the update scheme
and would therefore hide the very differences these tests must catch.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynem.core import NEM
from pynem.ppanggolin import (
    build_param_init,
    build_nem_neighborhood,
    partition_pangenome,
)

# Reference implementation (skip everything if unavailable)
run_partitioning = pytest.importorskip(
    "ppanggolin.nem.partition"
).run_partitioning

K = 3
SM_DEGREE = 10
BETA_RAW = 2.5


# ── Simulated pangenome (module-scoped, deterministic) ───────────────────────

@pytest.fixture(scope="module")
def pangenome():
    """A small simulated pangenome with weighted contiguity edges."""
    rng = np.random.default_rng(7)
    g_org = 30
    n_pers, n_shell, n_cloud = 40, 60, 80
    labels_true = np.array([0] * n_pers + [1] * n_shell + [2] * n_cloud)
    n = len(labels_true)

    def presence(lab):
        if lab == 0:
            p = rng.uniform(0.85, 1.0)
        elif lab == 1:
            p = rng.uniform(0.3, 0.7)
        else:
            p = rng.uniform(0.02, 0.18)
        return (rng.random(g_org) < p).astype(int)

    X = np.vstack([presence(lab) for lab in labels_true]).astype(float)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for c in range(3):
        idx = np.where(labels_true == c)[0]
        for a, b in zip(idx, np.roll(idx, 1)):
            graph.add_edge(int(a), int(b), weight=round(rng.uniform(0.2, 1.0), 4))
        for _ in range(len(idx)):
            a, b = rng.choice(idx, 2, replace=False)
            if a != b:
                graph.add_edge(int(a), int(b), weight=round(rng.uniform(0.2, 1.0), 4))
    for _ in range(20):
        a, b = map(int, rng.integers(0, n, 2))
        if a != b:
            graph.add_edge(a, b, weight=round(rng.uniform(0.2, 1.0), 4))

    H, total_edge_weight = build_nem_neighborhood(graph, sm_degree=SM_DEGREE)
    beta_scaled = BETA_RAW * (n / total_edge_weight)
    truth = np.array(["P", "S", "C"])[labels_true]
    return dict(X=X, graph=graph, H=H, n=n, g_org=g_org,
                beta_scaled=beta_scaled, truth=truth)


def _write_nem_files(d, X, graph, sm_degree):
    """Write NEM input files matching ``build_nem_neighborhood`` exactly."""
    d.mkdir(parents=True, exist_ok=True)
    n, g = X.shape
    (d / "nem_file.str").write_text(f"S\t{n}\t{g}\n")
    with open(d / "nem_file.dat", "w") as f:
        for row in X:
            f.write("\t".join(str(int(v)) for v in row) + "\n")
    with open(d / "nem_file.index", "w") as f:
        for i in range(n):
            f.write(f"{i + 1}\tfam_{i}\n")
    with open(d / "nem_file.nei", "w") as f:
        f.write("1\n")
        for i in range(n):
            nbrs = list(graph.neighbors(i))
            if 0 < len(nbrs) < sm_degree:
                ws = [graph[i][j].get("weight", 1.0) for j in nbrs]
                row = [i + 1, len(nbrs)] + [j + 1 for j in nbrs] \
                    + [round(w, 4) for w in ws]
                f.write("\t".join(map(str, row)) + "\n")
            else:
                f.write(f"{i + 1}\t0\n")


def _beta_for(pg, sm_degree):
    _, tew = build_nem_neighborhood(pg["graph"], sm_degree=sm_degree)
    return BETA_RAW * (pg["n"] / tew)


def _run_reference(pg, itermax, sm_degree=SM_DEGREE):
    """Run PPanGGOLiN's NEM; return (membership (N,K), (mu, eps, prop), part)."""
    tmp = Path(tempfile.mkdtemp(prefix="pynem_ref_"))
    _write_nem_files(tmp, pg["X"], pg["graph"], sm_degree)
    part, _, _ = run_partitioning(
        tmp, nb_org=pg["g_org"], beta=_beta_for(pg, sm_degree), kval=K, seed=42,
        init="param_file", itermax=itermax,
    )
    uf = np.loadtxt(tmp / f"nem_file_{K}.uf")
    lines = (tmp / f"nem_file_{K}.mf").read_text().splitlines()
    g = pg["g_org"]
    mu = np.zeros((K, g)); eps = np.zeros((K, g)); prop = np.zeros(K)
    for k, line in enumerate(lines[-K:]):
        v = line.split()
        mu[k] = [float(x) for x in v[:g]]
        prop[k] = float(v[g])
        eps[k] = [float(x) for x in v[g + 1:]]
    return uf, (mu, eps, prop), part


def _run_pynem(pg, itermax, site_update="seq", sm_degree=SM_DEGREE):
    H, _ = build_nem_neighborhood(pg["graph"], sm_degree=sm_degree)
    init_params = build_param_init(K, pg["g_org"])
    m = NEM(n_clusters=K, beta=_beta_for(pg, sm_degree), family="bernoulli",
            dispersion="sk_", proportion="pk", algorithm="nem",
            init="param", init_params=init_params, site_update=site_update,
            convergence="classification", tol=1e-9, missing="ignore",
            max_iter=itermax)
    m.fit(pg["X"], graph=H)
    return m


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("itermax", [1, 2, 5, 30])
def test_membership_matches_reference(pangenome, itermax):
    """Sequential NEM matches PPanGGOLiN's membership element-wise."""
    ref, _, _ = _run_reference(pangenome, itermax)
    got = _run_pynem(pangenome, itermax, "seq").membership_
    # ~1e-3: float32 C core + 6-decimal .uf + median tie handling
    assert np.max(np.abs(got - ref)) < 5e-3


def test_hub_asymmetry_matches_reference(pangenome):
    """With a low sm_degree most families become hubs/isolated, exercising the
    asymmetric neighbourhood (a hub is written isolated, yet still listed by its
    lower-degree neighbours). This confirms NEM does *not* symmetrise the .nei
    and that the directed-graph model is correct on exactly the structure that
    persistent (hub) gene families have."""
    sm = 3
    n_hub = sum(1 for i in range(pangenome["n"])
                if not (0 < pangenome["graph"].degree(i) < sm))
    assert n_hub > pangenome["n"] // 2          # the asymmetric path is exercised
    ref, _, _ = _run_reference(pangenome, 30, sm_degree=sm)
    got = _run_pynem(pangenome, 30, "seq", sm_degree=sm).membership_
    assert np.max(np.abs(got - ref)) < 1e-2


def test_parallel_update_does_not_match_early(pangenome):
    """Guard: the comparison is sensitive — Jacobi (parallel) breaks the
    per-iteration trajectory, so a future regression to parallel is caught."""
    ref, _, _ = _run_reference(pangenome, 1)
    par = _run_pynem(pangenome, 1, "parallel").membership_
    assert np.max(np.abs(par - ref)) > 0.1


def test_parameters_match_reference(pangenome):
    """Modes (mu), dispersions (epsilon) and proportions match the .mf file."""
    _, (mu_ref, eps_ref, prop_ref), _ = _run_reference(pangenome, 30)
    m = _run_pynem(pangenome, 30, "seq")
    assert np.all(np.isin(m.centers_, [0.0, 1.0]))         # binary modes
    assert np.max(np.abs(m.centers_ - mu_ref)) == 0.0       # exact mode match
    assert np.max(np.abs(m.dispersions_ - eps_ref)) < 1e-3
    assert np.max(np.abs(m.proportions_ - prop_ref)) < 1e-2


def test_mean_center_would_not_match(pangenome):
    """Guard: the median (mode) center is required — a mean center diverges."""
    _, (mu_ref, _, _), _ = _run_reference(pangenome, 30)
    m = _run_pynem(pangenome, 30, "seq")
    C, X = m.membership_, pangenome["X"]
    mean_centers = (C.T @ X) / np.maximum(C.sum(0)[:, None], 1e-12)
    assert np.max(np.abs(mean_centers - mu_ref)) > 0.1
    assert not np.all(np.isin(np.round(mean_centers, 6), [0.0, 1.0]))


def test_partition_pangenome_end_to_end(pangenome):
    """The public pipeline agrees with PPanGGOLiN's P/S/C labels and the truth."""
    _, _, ref_part = _run_reference(pangenome, 100)
    ref_lab = np.array(["P" if ref_part[f"fam_{i}"] == "P"
                        else "C" if ref_part[f"fam_{i}"] == "C"
                        else "S" for i in range(pangenome["n"])])

    res = partition_pangenome(pangenome["X"], pangenome["graph"],
                              K=K, beta=BETA_RAW, sm_degree=SM_DEGREE)
    assert res["partition"].shape == (pangenome["n"],)
    assert np.all(np.isin(res["partition"], ["P", "S", "C"]))
    # near-perfect agreement with the reference and with the simulated truth
    assert np.mean(res["partition"] == ref_lab) > 0.98
    assert np.mean(res["partition"] == pangenome["truth"]) > 0.95
