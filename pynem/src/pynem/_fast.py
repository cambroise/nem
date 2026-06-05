"""Optional Numba-accelerated kernels.

The sequential (Gauss-Seidel) E-step cannot be vectorised across nodes — each
node reads the already-updated memberships of its earlier neighbours. This loop
is JIT-compiled with Numba when available; otherwise the pure-Python loop in
``core.py`` is used. The two paths are numerically identical (float64, same
operations and visit order), so PPanGGOLiN reproduction is unaffected.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional dependency
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op fallback so the module imports without numba."""
        if args and callable(args[0]):
            return args[0]

        def wrap(func):
            return func

        return wrap


@njit(cache=True)
def seq_sweep(CM, log_pkfki, indptr, indices, data, beta, harden):
    """One in-place sequential (Gauss-Seidel) E-step sweep.

    Parameters
    ----------
    CM : (N, K) float64 — classification, updated in place and returned.
    log_pkfki : (N, K) float64 — precomputed log(p_k f_k(x_i)).
    indptr, indices, data : CSR arrays of the (directed) weighted adjacency.
    beta : float — spatial smoothing strength.
    harden : bool — if True apply the C-step (argmax -> one-hot), i.e. NCEM.

    Mirrors ``NEM._normalize_local`` / ``_harden_node`` for the mean-field and
    ICM (ncem) variants. Gibbs (gem) needs an RNG and stays on the Python path.
    """
    N = CM.shape[0]
    K = CM.shape[1]
    ctx = np.empty(K)
    lognum = np.empty(K)

    for i in range(N):
        # spatial context: sum_{j in N(i)} w_ij * CM[j]
        for k in range(K):
            ctx[k] = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            w = data[p]
            for k in range(K):
                ctx[k] += w * CM[j, k]

        # log numerator and its max
        maxlog = -np.inf
        for k in range(K):
            val = log_pkfki[i, k] + beta * ctx[k]
            lognum[k] = val
            if val > maxlog:
                maxlog = val

        if not np.isfinite(maxlog):
            for k in range(K):
                CM[i, k] = 1.0 / K
            continue

        total = 0.0
        for k in range(K):
            e = np.exp(lognum[k] - maxlog)
            lognum[k] = e
            total += e

        if total <= 0.0:
            for k in range(K):
                CM[i, k] = 1.0 / K
            continue

        if harden:
            best = 0
            bestval = lognum[0]
            for k in range(1, K):
                if lognum[k] > bestval:
                    bestval = lognum[k]
                    best = k
            for k in range(K):
                CM[i, k] = 1.0 if k == best else 0.0
        else:
            for k in range(K):
                CM[i, k] = lognum[k] / total

    return CM
