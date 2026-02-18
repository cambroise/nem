"""Graph/neighborhood handling for NEM."""

import numpy as np
import networkx as nx


class NeighborhoodSystem:
    """Efficient adjacency structure extracted from a networkx graph.

    Parameters
    ----------
    G : nx.Graph
        Graph with optional 'weight' edge attribute (default 1.0).
    """

    def __init__(self, G):
        self._n = G.number_of_nodes()
        # Build adjacency lists with weights
        self._neighbors = []
        self._max_neighbors = 0
        for i in range(self._n):
            neighs = []
            for j in G.neighbors(i):
                w = G.edges[i, j].get("weight", 1.0)
                neighs.append((j, w))
            self._neighbors.append(neighs)
            self._max_neighbors = max(self._max_neighbors, len(neighs))

    @property
    def n_nodes(self):
        return self._n

    @property
    def max_neighbors(self):
        return self._max_neighbors

    def neighbors(self, i):
        """Return list of (index, weight) for neighbors of node i."""
        return self._neighbors[i]

    def spatial_context(self, i, C, K):
        """Compute spatial context for node i.

        Returns
        -------
        context : array of shape (K,)
            context[k] = sum_{j in N(i)} w_ij * c_jk
        """
        context = np.zeros(K)
        for j, w in self._neighbors[i]:
            context += w * C[j]
        return context

    def compute_all_contexts(self, C):
        """Compute spatial context for all nodes at once.

        Parameters
        ----------
        C : (N, K) array — classification matrix.

        Returns
        -------
        contexts : (N, K) array
            contexts[i, k] = sum_{j in N(i)} w_ij * c_jk
        """
        N, K = C.shape
        contexts = np.zeros((N, K))
        for i in range(N):
            for j, w in self._neighbors[i]:
                contexts[i] += w * C[j]
        return contexts

    def compute_G(self, C):
        """Compute geographic cohesion criterion.

        G = sum_i sum_k c_ik * context_ik
        """
        contexts = self.compute_all_contexts(C)
        return (C * contexts).sum()
