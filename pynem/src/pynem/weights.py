"""Derive per-variable weights for the weighted NEM from group structure.

The weighted NEM (see :class:`pynem.NEM` ``feature_weights`` /
:func:`pynem.partition_pangenome` ``genome_weights``) down-weights redundant
variables. A common source of redundancy in a pangenome is an **over-sampled
clade**: several near-identical genomes of the same species. This module derives
the weights automatically by grouping the genomes and applying the
abundance-correcting rule of the weighted-NEM note,

    w_j = (d / m) * (1 / |G_{g(j)}|),

where ``d`` is the number of genomes, ``m`` the number of groups and ``G_{g(j)}``
the group of genome ``j``. Each group then contributes ``d/m`` in total
*regardless of its size*, so a clade of 50 strains no longer outweighs a lone
genome (this is the sequence-weighting idea of Henikoff & Henikoff 1994). The
weights sum to ``d`` (mean weight 1), matching the "variable replication"
interpretation of the note.

The grouping uses hierarchical clustering with **Jaccard distance** and **UPGMA**
(average linkage), via scipy. The number of groups is either fixed (default 10)
or chosen by the **silhouette** criterion (no scikit-learn dependency; the
silhouette is computed on the same Jaccard distances used for the linkage).
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "jaccard_upgma_linkage",
    "silhouette_score",
    "jaccard_upgma_labels",
    "abundance_weights",
    "genome_weights",
]


def jaccard_upgma_linkage(observations):
    """Jaccard distances (condensed) and the UPGMA linkage over the rows.

    Parameters
    ----------
    observations : (n_samples, n_features) array-like
        Binary presence/absence rows. Cast to bool for the Jaccard metric.

    Returns
    -------
    condensed : (n_samples*(n_samples-1)/2,) array
        Condensed pairwise Jaccard distances (``scipy.spatial.distance.pdist``).
    Z : (n_samples-1, 4) array
        UPGMA linkage matrix (``method='average'``).
    """
    obs = np.asarray(observations)
    if obs.ndim != 2:
        raise ValueError(f"observations must be 2-D; got shape {obs.shape}")
    if obs.shape[0] < 2:
        raise ValueError("need at least 2 samples to cluster")
    # Jaccard on bool: scipy's float behaviour is version-dependent -> cast.
    condensed = pdist(obs.astype(bool), metric="jaccard")
    # pdist may yield NaN for an all-zero pair (0/0); treat as distance 0.
    condensed = np.nan_to_num(condensed, nan=0.0)
    Z = linkage(condensed, method="average")  # UPGMA
    return condensed, Z


def silhouette_score(distances, labels):
    """Mean silhouette of a clustering, from a **precomputed** distance matrix.

    Parameters
    ----------
    distances : array-like
        Either a condensed distance vector (as from ``pdist``) or a square
        ``(n, n)`` distance matrix.
    labels : (n,) array-like
        Cluster id of each sample.

    Returns
    -------
    float
        Mean silhouette in ``[-1, 1]``. Returns ``-1.0`` for a degenerate
        clustering (fewer than 2 clusters, or as many clusters as samples).
        Singleton clusters contribute a silhouette of 0 (sklearn convention).
    """
    D = np.asarray(distances, dtype=float)
    if D.ndim == 1:
        D = squareform(D)
    labels = np.asarray(labels)
    n = len(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= n:
        return -1.0
    members = {c: np.where(labels == c)[0] for c in uniq}
    s = np.zeros(n)
    for i in range(n):
        same = members[labels[i]]
        if len(same) <= 1:
            continue  # singleton -> silhouette 0
        a = D[i, same].sum() / (len(same) - 1)  # self-distance is 0
        b = min(D[i, members[c]].mean() for c in uniq if c != labels[i])
        denom = max(a, b)
        s[i] = (b - a) / denom if denom > 0 else 0.0
    return float(s.mean())


def jaccard_upgma_labels(observations, n_clusters=10, selection="fixed",
                         k_range=None):
    """Cluster the rows of ``observations`` (Jaccard + UPGMA) into groups.

    Parameters
    ----------
    observations : (n_samples, n_features) array-like
        Binary rows to group (e.g. genomes-as-rows).
    n_clusters : int
        Number of groups when ``selection='fixed'`` (default 10). Capped at
        ``n_samples``.
    selection : {'fixed', 'silhouette'}
        ``'fixed'`` cuts the dendrogram into ``n_clusters`` groups.
        ``'silhouette'`` scans ``k_range`` and keeps the cut maximising the mean
        silhouette (computed on the Jaccard distances).
    k_range : iterable of int or None
        Candidate group counts for the silhouette search. Defaults to
        ``2 .. min(20, n_samples-1)``.

    Returns
    -------
    labels : (n_samples,) int array
        Group id of each sample (1-based, as returned by ``fcluster``).
    """
    obs = np.asarray(observations)
    n = obs.shape[0]
    condensed, Z = jaccard_upgma_linkage(obs)

    if selection == "fixed":
        k = max(1, min(int(n_clusters), n))
        return fcluster(Z, t=k, criterion="maxclust")

    if selection == "silhouette":
        if k_range is None:
            k_range = range(2, min(20, n - 1) + 1)
        Dsq = squareform(condensed)
        best_labels, best_score = None, -np.inf
        for k in k_range:
            if k < 2 or k >= n:
                continue
            lab = fcluster(Z, t=k, criterion="maxclust")
            if len(np.unique(lab)) < 2:
                continue
            sc = silhouette_score(Dsq, lab)
            if sc > best_score:
                best_score, best_labels = sc, lab
        if best_labels is None:  # no valid cut -> single group
            return np.ones(n, dtype=int)
        return best_labels

    raise ValueError(f"selection must be 'fixed' or 'silhouette'; got "
                     f"{selection!r}")


def abundance_weights(labels):
    """Inverse-abundance weights from a grouping: ``w_j = (d/m)·(1/|G_{g(j)}|)``.

    Parameters
    ----------
    labels : (d,) array-like
        Group id of each item (genome).

    Returns
    -------
    weights : (d,) array
        Positive weights summing to ``d`` (mean 1): every group contributes
        ``d/m`` in total, independently of its size.
    """
    labels = np.asarray(labels)
    d = len(labels)
    uniq, inverse, counts = np.unique(labels, return_inverse=True,
                                      return_counts=True)
    m = len(uniq)
    group_size = counts[inverse]            # |G_{g(j)}| for each j
    return (d / m) * (1.0 / group_size)


def genome_weights(presence, n_groups=10, selection="fixed", k_range=None,
                   return_labels=False):
    """Automatic per-genome weights for the weighted NEM on a pangenome.

    Groups the **genomes** (columns of ``presence``) by Jaccard distance and
    UPGMA, then applies the inverse-abundance rule so an over-sampled clade does
    not dominate the partition. Feed the result to
    ``partition_pangenome(..., genome_weights=w)`` or ``NEM(feature_weights=w)``.

    Parameters
    ----------
    presence : (n_families, n_genomes) array-like of {0, 1}
        Gene-family presence/absence (families in rows, genomes in columns).
    n_groups : int
        Number of genome groups when ``selection='fixed'`` (default 10).
    selection : {'fixed', 'silhouette'}
        How to choose the number of groups.
    k_range : iterable of int or None
        Candidate group counts for the silhouette search.
    return_labels : bool
        If True, also return the genome group labels.

    Returns
    -------
    weights : (n_genomes,) array
        Per-genome weights summing to ``n_genomes``.
    labels : (n_genomes,) int array
        Only if ``return_labels`` — the genome group of each column.
    """
    presence = np.asarray(presence)
    if presence.ndim != 2:
        raise ValueError(f"presence must be 2-D (n_families, n_genomes); got "
                         f"{presence.shape}")
    labels = jaccard_upgma_labels(presence.T, n_clusters=n_groups,
                                  selection=selection, k_range=k_range)
    weights = abundance_weights(labels)
    if return_labels:
        return weights, labels
    return weights
