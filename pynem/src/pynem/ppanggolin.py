"""Reproduce the PPanGGOLiN pangenome-partitioning pipeline with pynem.

PPanGGOLiN (Gautreau et al. 2020) partitions gene families into
**persistent / shell / cloud** by running the NEM algorithm on a graph of gene
families: each node is a gene family carrying a binary presence/absence vector
across genomes (a multivariate Bernoulli feature), and edges encode genomic
contiguity. Its partitioning step is exactly the NEM algorithm reimplemented in
:mod:`pynem`.

This module wires :class:`pynem.NEM` to reproduce that pipeline with
PPanGGOLiN's default settings (``ppanggolin/ppanggolin/nem/partition.py``):

================  ===========================================================
Setting           Value
================  ===========================================================
algorithm         ``nem`` (mean-field)
family            ``bernoulli`` (centre = binary mode, via weighted median)
n_clusters        ``K = 3``  → class 0 persistent, K-1 cloud, middle shell
beta              ``2.5`` rescaled by ``nb_families / total_edge_weight``
proportion        ``pk`` (free)
dispersion        ``sk_`` (per class) — ``skd`` if ``free_dispersion``
site update       ``seq`` (sequential / Gauss-Seidel, NEM default)
missing           ``ignore`` (forced for Bernoulli in the C code)
convergence       classification change ``< 0.01``
initialization    structured parameter file (``build_param_init``)
================  ===========================================================

The smoothing-degree filter (``sm_degree``) is reproduced faithfully, including
its **asymmetry**: a family with at least ``sm_degree`` neighbours is excluded
from smoothing (written with no neighbours), yet still appears in the neighbour
lists of its lower-degree neighbours. This is modelled with a directed graph.
"""

import math

import networkx as nx
import numpy as np

from .core import NEM


def build_param_init(K, n_org):
    """Structured parameter initialization used by PPanGGOLiN.

    Port of the ``nem_file_init_*.m`` file written by ``run_partitioning``
    (``partition.py``). Class 0 has mode 1 (present everywhere → seeds the
    persistent partition), the upper classes have mode 0, and dispersions are
    laid out in a staircase.

    Parameters
    ----------
    K : int
        Number of partitions.
    n_org : int
        Number of genomes (feature dimension D).

    Returns
    -------
    centers : (K, n_org) array
        Initial modes (mu), 0 or 1.
    dispersions : (K, n_org) array
        Initial dispersions (epsilon).
    proportions : (K,) array
        Initial proportions (≈ 1/K, with the same 2-decimal rounding as NEM).
    """
    # Proportions: K-1 values rounded to 2 decimals, last one by subtraction.
    p = round(1.0 / K, 2)
    proportions = np.array([p] * (K - 1) + [1.0 - p * (K - 1)], dtype=float)

    centers = np.zeros((K, n_org))
    dispersions = np.zeros((K, n_org))
    step = 0.5 / math.ceil(K / 2)
    pichenette = 0.1 if K == 2 else 0.0
    for k in range(1, K + 1):
        if k <= K / 2:
            mu, eps = 1.0, step * k - pichenette
        else:
            mu, eps = 0.0, step * (K - k + 1) - pichenette
        centers[k - 1, :] = mu
        dispersions[k - 1, :] = eps
    return centers, dispersions, proportions


def build_nem_neighborhood(graph, sm_degree=10):
    """Build the directed NEM neighbourhood and the beta-rescaling weight.

    Reproduces ``write_nem_input_files``: a family with ``>= sm_degree``
    neighbours (or none) is written isolated, otherwise all its weighted
    neighbours are written. The result is asymmetric, hence a ``DiGraph``.

    Parameters
    ----------
    graph : nx.Graph
        Gene-family contiguity graph; edge weights (default 1.0) are the
        ``distance_score`` (coverage / n_org) of PPanGGOLiN.
    sm_degree : int
        Maximum degree of a node to be included in smoothing.

    Returns
    -------
    H : nx.DiGraph
        Directed neighbourhood consumed by :class:`pynem.NEM`.
    total_edge_weight : float
        ``sum_of_written_edge_weights / 2`` — used to rescale beta.
    """
    H = nx.DiGraph()
    H.add_nodes_from(graph.nodes())
    total = 0.0
    for i in graph.nodes():
        nbrs = list(graph.neighbors(i))
        if 0 < len(nbrs) < sm_degree:
            for j in nbrs:
                w = graph[i][j].get("weight", 1.0)
                H.add_edge(i, j, weight=w)
                total += w
        # else: hub or isolated family -> no outgoing smoothing edges
    return H, total / 2.0


def _assign_partition(c_i, K):
    """PPanGGOLiN label for one family from its membership vector.

    Returns 'P', 'C' or 'S' (shell). A family is shell either because its top
    class is a middle class, or "in case of doubt" — a tie on the maximum, or a
    maximum probability below 0.5.
    """
    mx = c_i.max()
    args = np.flatnonzero(c_i == mx)
    if len(args) > 1 or mx < 0.5:
        return "S"
    k = args[0]
    if k == 0:
        return "P"
    if k == K - 1:
        return "C"
    return "S"


def partition_pangenome(presence, graph, K=3, beta=2.5, free_dispersion=False,
                        sm_degree=10, max_iter=100, tol=0.01,
                        genome_weights=None, completeness=None):
    """Partition gene families into persistent / shell / cloud, à la PPanGGOLiN.

    Parameters
    ----------
    presence : (N, n_org) array of {0, 1}
        Presence/absence of each of the N gene families across n_org genomes.
    graph : nx.Graph
        Gene-family contiguity graph (optionally weighted via 'weight').
    K : int
        Number of partitions (default 3: persistent / shell / cloud).
    beta : float
        Smoothing strength before rescaling (PPanGGOLiN default 2.5).
        Internally rescaled by ``N / total_edge_weight``.
    free_dispersion : bool
        If True use the ``skd`` model (per class and genome), else ``sk_``.
    sm_degree : int
        Smoothing-degree filter (default 10).
    max_iter : int
        Maximum NEM iterations.
    tol : float
        Classification-change convergence threshold (PPanGGOLiN uses 0.01).
    genome_weights : (n_org,) array-like or None
        Per-genome (per-column) weights ``w_j > 0`` for the *weighted* NEM,
        e.g. to down-weight redundant genomes from an over-sampled clade so the
        partition reflects biology rather than sampling. ``None`` (default) =
        all weights 1, the standard PPanGGOLiN-faithful behaviour.
    completeness : (n_org,) array-like, ``"auto"`` or None
        Per-genome completeness ``gamma_j in (0, 1]`` (e.g. from CheckM) for the
        **MAG-aware** Bernoulli model (mOTUpan-style). An absence in an
        incomplete genome is forgiven (presence prob ``mu * gamma_j``), which
        stops incomplete metagenome-assembled genomes from artificially
        shrinking the persistent class. ``"auto"`` self-estimates completeness
        from the data (no CheckM needed): length-seed init then iterative
        re-estimation from the inferred persistent set (mOTUpan Eq. 6); the
        estimated vector is returned in ``result["completeness"]``. ``None``
        (default) = full completeness, the standard behaviour.

    Returns
    -------
    dict with keys:
        ``partition``      (N,) array of 'P'/'S'/'C'
        ``membership``     (N, K) soft classification
        ``labels_index``   (N,) hard class index 1..K (0=persistent … K-1=cloud)
        ``centers``        (K, n_org) modes mu (binary)
        ``dispersions``    (K, n_org) dispersions epsilon
        ``proportions``    (K,) class proportions
        ``beta``           rescaled beta actually used
        ``criteria``       dict of NEM criteria (U, D, G, L, M)
        ``n_iter``         number of iterations run
        ``model``          the fitted :class:`pynem.NEM` instance
    """
    presence = np.asarray(presence, dtype=float)
    N, n_org = presence.shape

    # completeness="auto": estimate per-genome completeness from the data itself
    # (no CheckM) by alternating partition <-> Eq.6 re-estimation.
    if isinstance(completeness, str):
        if completeness != "auto":
            raise ValueError("completeness must be None, an array, or 'auto'; "
                             f"got {completeness!r}")
        return _partition_self_completeness(
            presence, graph, K=K, beta=beta, free_dispersion=free_dispersion,
            sm_degree=sm_degree, max_iter=max_iter, tol=tol,
            genome_weights=genome_weights)

    H, total_edge_weight = build_nem_neighborhood(graph, sm_degree=sm_degree)
    if total_edge_weight <= 0:
        beta_scaled = beta
    else:
        beta_scaled = beta * (N / total_edge_weight)

    init_params = build_param_init(K, n_org)

    model = NEM(
        n_clusters=K,
        beta=beta_scaled,
        family="bernoulli",
        dispersion="skd" if free_dispersion else "sk_",
        proportion="pk",
        algorithm="nem",
        init="param",
        init_params=init_params,
        site_update="seq",
        feature_weights=genome_weights,
        completeness=completeness,
        convergence="classification",
        tol=tol,
        missing="ignore",
        max_iter=max_iter,
    )
    model.fit(presence, graph=H)

    partition = np.array(
        [_assign_partition(model.membership_[i], K) for i in range(N)]
    )

    return {
        "partition": partition,
        "membership": model.membership_,
        "labels_index": model.labels_,
        "centers": model.centers_,
        "dispersions": model.dispersions_,
        "proportions": model.proportions_,
        "beta": beta_scaled,
        "criteria": model.criteria_,
        "n_iter": model.n_iter_,
        "completeness": completeness,
        "model": model,
    }


def _partition_self_completeness(presence, graph, K=3, beta=2.5,
                                 free_dispersion=False, sm_degree=10,
                                 max_iter=100, tol=0.01, genome_weights=None,
                                 comp_iters=15, comp_tol=1e-3, gmin=0.05):
    """Self-estimate per-genome completeness (mOTUpan Eq. 6), no CheckM.

    Initialise ``gamma_j`` from a genome-size proxy (a more complete genome has
    more genes), then alternate: partition with ``completeness=gamma`` -> take
    the persistent set ``P`` -> re-estimate ``gamma_j = |P present in j| / |P|``
    -> repeat until ``gamma`` stabilises. Returns the final
    :func:`partition_pangenome` result dict, with ``completeness`` set to the
    estimated vector and ``completeness_n_iter`` to the number of outer rounds.
    """
    presence = np.asarray(presence, dtype=float)
    sizes = presence.sum(axis=0)
    gamma = np.clip(sizes / max(sizes.max(), 1.0), gmin, 1.0)   # length-seed init
    res = None
    n_round = 0
    for n_round in range(1, comp_iters + 1):
        res = partition_pangenome(
            presence, graph, K=K, beta=beta, free_dispersion=free_dispersion,
            sm_degree=sm_degree, max_iter=max_iter, tol=tol,
            genome_weights=genome_weights, completeness=gamma)
        persistent = res["partition"] == "P"
        if persistent.sum() == 0:
            break
        new_gamma = np.clip(presence[persistent].sum(axis=0) / persistent.sum(),
                            gmin, 1.0)                            # Eq. 6
        delta = float(np.max(np.abs(new_gamma - gamma)))
        gamma = new_gamma
        if delta < comp_tol:
            break
    res["completeness"] = gamma
    res["completeness_n_iter"] = n_round
    return res
