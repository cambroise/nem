"""Graph/neighborhood handling for NEM.

The neighborhood is stored once as a sparse CSR adjacency matrix ``A`` where
``A[i, j]`` is the weight ``w_ij`` of edge ``i -> j`` as listed for node ``i``
(the matrix is generally **not** symmetric — e.g. PPanGGOLiN's ``sm_degree``
hub filter produces a directed neighborhood). The spatial context of all nodes
is then the matrix product ``A @ C``:

    context[i, k] = sum_{j in N(i)} w_ij * c_jk

The same CSR arrays (``indptr``/``indices``/``data``) also back the per-node
access used by the sequential (Gauss-Seidel) E-step.
"""

import numpy as np
import scipy.sparse as sp


class NeighborhoodSystem:
    """Sparse adjacency structure extracted from a networkx graph.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Graph on nodes ``0..n-1`` with optional 'weight' edge attribute
        (default 1.0). For a directed graph, row ``i`` of the adjacency lists
        the successors of ``i``.
    """

    def __init__(self, G):
        n = G.number_of_nodes()
        self._n = n

        rows, cols, data = [], [], []
        for i in range(n):
            for j in G.neighbors(i):
                rows.append(i)
                cols.append(j)
                data.append(G.edges[i, j].get("weight", 1.0))

        self.A = sp.csr_matrix(
            (np.asarray(data, dtype=float),
             (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(n, n),
        )
        # Cached CSR arrays for fast per-node access (Gauss-Seidel sweep).
        self._indptr = self.A.indptr
        self._indices = self.A.indices
        self._data = self.A.data
        self._max_neighbors = (
            int(np.diff(self.A.indptr).max()) if n > 0 else 0
        )

    @property
    def n_nodes(self):
        return self._n

    @property
    def max_neighbors(self):
        return self._max_neighbors

    def neighbors(self, i):
        """Return list of (index, weight) for the neighbors of node i."""
        s, e = self._indptr[i], self._indptr[i + 1]
        return list(zip(self._indices[s:e].tolist(), self._data[s:e].tolist()))

    def spatial_context(self, i, C, K):
        """Spatial context for node i: ``sum_{j in N(i)} w_ij * C[j]``.

        Returns
        -------
        context : array of shape (K,)
        """
        s, e = self._indptr[i], self._indptr[i + 1]
        if e == s:
            return np.zeros(K)
        return self._data[s:e] @ C[self._indices[s:e]]

    def compute_all_contexts(self, C):
        """Spatial context for all nodes at once: ``A @ C``.

        Parameters
        ----------
        C : (N, K) array — classification matrix.

        Returns
        -------
        contexts : (N, K) array
            ``contexts[i, k] = sum_{j in N(i)} w_ij * c_jk``.
        """
        return self.A @ C

    def compute_G(self, C):
        """Geographic cohesion criterion ``G = sum_i sum_k c_ik * context_ik``."""
        return float((C * (self.A @ C)).sum())
