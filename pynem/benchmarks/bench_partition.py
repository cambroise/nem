#!/usr/bin/env python3
"""Benchmark / profile pynem's NEM partitioning.

Captures the performance work done while optimising pynem (sparse CSR contexts,
vectorised density, threshold Bernoulli centre, Numba sequential E-step).

Usage
-----
    python benchmarks/bench_partition.py            # default sizes
    python benchmarks/bench_partition.py --profile  # cProfile hot spots
    python benchmarks/bench_partition.py -n 22000 -d 25 --iters 100

It reports wall-clock for the Numba and pure-Python sequential E-step and
checks that they agree (they should, to ~1e-12), then optionally prints a
cProfile breakdown. The speedup grows when the neighbourhood loop dominates
(small D / dense graph / many iterations) and shrinks when the vectorised
density dominates (large D).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

# Make pynem importable when run from the repo without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pynem.core as core
from pynem.core import NEM
from pynem.ppanggolin import build_param_init, build_nem_neighborhood
from pynem._fast import HAS_NUMBA


def simulate(n_pers, n_shell, n_cloud, d, density, seed=1):
    """Simulate a pangenome: presence/absence matrix + contiguity graph."""
    rng = np.random.default_rng(seed)
    labels = np.array([0] * n_pers + [1] * n_shell + [2] * n_cloud)
    n = len(labels)

    def presence(lab):
        if lab == 0:
            p = rng.uniform(0.85, 1.0)
        elif lab == 1:
            p = rng.uniform(0.3, 0.7)
        else:
            p = rng.uniform(0.02, 0.18)
        return (rng.random(d) < p).astype(int)

    X = np.vstack([presence(lab) for lab in labels]).astype(float)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for c in range(3):
        idx = np.where(labels == c)[0]
        for a, b in zip(idx, np.roll(idx, 1)):
            G.add_edge(int(a), int(b), weight=1.0)
        for _ in range(density * len(idx)):
            a, b = rng.choice(idx, 2, replace=False)
            if a != b:
                G.add_edge(int(a), int(b), weight=1.0)
    return X, G, labels


def make_model(X, H, beta, max_iter):
    return NEM(n_clusters=3, beta=beta, family="bernoulli", dispersion="sk_",
               proportion="pk", algorithm="nem", init="param",
               init_params=build_param_init(3, X.shape[1]), site_update="seq",
               convergence="classification", tol=1e-4, missing="ignore",
               max_iter=max_iter)


def time_fit(X, H, beta, max_iter, use_numba):
    saved = core.HAS_NUMBA
    core.HAS_NUMBA = use_numba and HAS_NUMBA
    try:
        t = time.perf_counter()
        m = make_model(X, H, beta, max_iter).fit(X, graph=H)
        dt = time.perf_counter() - t
    finally:
        core.HAS_NUMBA = saved
    return m, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-families", type=int, default=22000)
    ap.add_argument("-d", "--genomes", type=int, default=40)
    ap.add_argument("--density", type=int, default=3,
                    help="intra-class random edges per node")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--sm-degree", type=int, default=10)
    ap.add_argument("--profile", action="store_true",
                    help="print a cProfile breakdown of a single fit")
    args = ap.parse_args()

    # Split N across the three partitions ~ 15% / 35% / 50%.
    n = args.n_families
    X, G, _ = simulate(int(0.15 * n), int(0.35 * n), n - int(0.15 * n) - int(0.35 * n),
                       args.genomes, args.density)
    H, tew = build_nem_neighborhood(G, sm_degree=args.sm_degree)
    beta = 2.5 * (X.shape[0] / tew)
    print(f"N={X.shape[0]}  D={args.genomes}  edges={G.number_of_edges()}  "
          f"numba={'yes' if HAS_NUMBA else 'NO (pure Python)'}")

    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        m, _ = time_fit(X, H, beta, args.iters, use_numba=True)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
        print(f"n_iter={m.n_iter_}\n{s.getvalue()}")
        return

    if HAS_NUMBA:
        time_fit(X, H, beta, args.iters, use_numba=True)          # warm up JIT
    m_nb, t_nb = time_fit(X, H, beta, args.iters, use_numba=True)
    m_py, t_py = time_fit(X, H, beta, args.iters, use_numba=False)
    diff = float(np.max(np.abs(m_nb.membership_ - m_py.membership_)))
    print(f"n_iter={m_nb.n_iter_}")
    print(f"  Python seq E-step : {t_py:6.2f}s")
    if HAS_NUMBA:
        print(f"  Numba  seq E-step : {t_nb:6.2f}s   speedup {t_py / t_nb:4.1f}x")
        print(f"  max|membership Numba - Python| = {diff:.1e}")


if __name__ == "__main__":
    main()
