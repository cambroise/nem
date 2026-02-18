"""Visualization utilities for NEM results."""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph_clusters(G, labels, pos=None, ax=None, **kwargs):
    """Draw graph with nodes colored by cluster label.

    Parameters
    ----------
    G : nx.Graph
    labels : array-like of shape (N,)
        Cluster labels (1-based).
    pos : dict or None
        Node positions. If None, uses spring layout.
    ax : matplotlib Axes or None
    **kwargs : passed to nx.draw_networkx_nodes.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    labels = np.asarray(labels)
    K = int(labels.max())
    cmap = plt.cm.get_cmap("tab10", K)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=labels - 1,
        cmap=cmap,
        vmin=0, vmax=K - 1,
        node_size=kwargs.get("node_size", 60),
        alpha=kwargs.get("alpha", 0.9),
    )
    ax.set_title("Graph clusters")
    ax.axis("off")
    return ax


def plot_membership(G, membership, class_idx=0, pos=None, ax=None):
    """Heatmap of soft membership for one class.

    Parameters
    ----------
    G : nx.Graph
    membership : (N, K) array
    class_idx : int
        Which class to display.
    pos : dict or None
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=membership[:, class_idx],
        cmap=plt.cm.viridis,
        vmin=0, vmax=1,
        node_size=60,
    )
    plt.colorbar(nodes, ax=ax, label=f"P(class {class_idx + 1})")
    ax.set_title(f"Membership — class {class_idx + 1}")
    ax.axis("off")
    return ax


def plot_convergence(history, criterion="U", ax=None):
    """Plot criterion value vs iteration.

    Parameters
    ----------
    history : list of dict
        Each dict has keys 'U', 'D', 'G', 'L', 'M'.
    criterion : str
        Which criterion to plot.
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    values = [h[criterion] for h in history]
    ax.plot(range(1, len(values) + 1), values, "o-", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(criterion)
    ax.set_title(f"Convergence — {criterion}")
    ax.grid(True, alpha=0.3)
    return ax


def plot_cluster_centers(centers, labels=None, ax=None):
    """Parallel coordinates plot of cluster centers.

    Parameters
    ----------
    centers : (K, D) array
    labels : list of str or None
        Variable names.
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    K, D = centers.shape
    x = np.arange(D)
    cmap = plt.cm.get_cmap("tab10", K)

    for k in range(K):
        ax.plot(x, centers[k], "o-", color=cmap(k), label=f"Class {k + 1}",
                markersize=6)

    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f"Var {d + 1}" for d in range(D)])

    ax.set_ylabel("Center value")
    ax.set_title("Cluster centers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
