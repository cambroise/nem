"""Optional Numba-accelerated kernels.

The sequential (Gauss-Seidel) E-step cannot be vectorised across nodes — each
node reads the already-updated memberships of its earlier neighbours. This loop
is JIT-compiled with Numba when available; otherwise the pure-Python loop in
``core.py`` is used. The two paths are numerically identical (float64, same
operations and visit order), so PPanGGOLiN reproduction is unaffected.
"""

import math

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional dependency
    HAS_NUMBA = False
    prange = range

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


_PROB_FLOOR = 1e-20
_ZERO_DISP_TOL = 1e-20


@njit(cache=True)
def density_normal(Xf, observed, centers, dispersions, proportions,
                    weights, has_weights, log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    log_coef = np.empty(D)
    for k in range(K):
        log_pk = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            if v > _ZERO_DISP_TOL:
                log_coef[d] = -0.5 * math.log(2.0 * math.pi * v)
        for i in range(N):
            log_fki = 0.0
            invalid = False
            for d in range(D):
                if not observed[i, d]:
                    continue
                v = dispersions[k, d]
                diff = Xf[i, d] - centers[k, d]
                if v <= _ZERO_DISP_TOL:
                    if abs(diff) > _ZERO_DISP_TOL:
                        if (not has_weights) or weights[d] > 0.0:
                            invalid = True
                            break
                    continue
                w = weights[d] if has_weights else 1.0
                term = log_coef[d] - 0.5 * diff * diff / v
                log_fki += term * w
            log_pkfki[i, k] = -np.inf if invalid else log_pk + log_fki


@njit(cache=True)
def density_laplace(Xf, observed, centers, dispersions, proportions,
                     weights, has_weights, log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    neg_log_2v = np.empty(D)
    for k in range(K):
        log_pk = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            if v > _ZERO_DISP_TOL:
                neg_log_2v[d] = -math.log(2.0 * v)
        for i in range(N):
            log_fki = 0.0
            invalid = False
            for d in range(D):
                if not observed[i, d]:
                    continue
                v = dispersions[k, d]
                diff = Xf[i, d] - centers[k, d]
                if v <= _ZERO_DISP_TOL:
                    if abs(diff) > _ZERO_DISP_TOL:
                        if (not has_weights) or weights[d] > 0.0:
                            invalid = True
                            break
                    continue
                w = weights[d] if has_weights else 1.0
                term = neg_log_2v[d] - abs(diff) / v
                log_fki += term * w
            log_pkfki[i, k] = -np.inf if invalid else log_pk + log_fki


@njit(cache=True, parallel=True)
def density_bernoulli(Xf, observed, centers, dispersions, proportions,
                       weights, has_weights, log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    log_one_minus_v = np.empty(D)
    log_ratio = np.empty(D)
    for k in range(K):
        log_pk = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            if v > _ZERO_DISP_TOL:
                log_one_minus_v[d] = math.log(1.0 - v)
                log_ratio[d] = math.log((1.0 - v) / v)

        # each i writes only log_pkfki[i, k]
        for i in prange(N):
            log_fki = 0.0
            invalid = False
            for d in range(D):
                if not observed[i, d]:
                    continue
                v = dispersions[k, d]
                diff = Xf[i, d] - centers[k, d]
                if v <= _ZERO_DISP_TOL:
                    if abs(diff) > _ZERO_DISP_TOL:
                        if (not has_weights) or weights[d] > 0.0:
                            invalid = True
                            break
                    continue
                w = weights[d] if has_weights else 1.0
                term = log_one_minus_v[d] - abs(diff) * log_ratio[d]
                log_fki += term * w
            log_pkfki[i, k] = -np.inf if invalid else log_pk + log_fki


@njit(cache=True)
def density_bernoulli_mag(Xf, observed, centers, dispersions, proportions,
                          completeness, weights, has_weights, log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    log_one_minus_mu_eff = np.empty(D)
    log_diff = np.empty(D)
    for k in range(K):
        log_pk = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            safe_v = 0.5 if v <= _ZERO_DISP_TOL else v
            mu = (1.0 - safe_v) if centers[k, d] == 1.0 else safe_v
            mu_eff = mu * completeness[d]
            if mu_eff < _PROB_FLOOR:
                mu_eff = _PROB_FLOOR
            elif mu_eff > 1.0 - _PROB_FLOOR:
                mu_eff = 1.0 - _PROB_FLOOR
            log_mu_eff = math.log(mu_eff)
            lome = math.log(1.0 - mu_eff)
            log_one_minus_mu_eff[d] = lome
            log_diff[d] = log_mu_eff - lome
        for i in range(N):
            log_fki = 0.0
            for d in range(D):
                if not observed[i, d]:
                    continue
                w = weights[d] if has_weights else 1.0
                term = log_one_minus_mu_eff[d] + Xf[i, d] * log_diff[d]
                log_fki += term * w
            log_pkfki[i, k] = log_pk + log_fki


@njit(cache=True, parallel=True)
def inertia_accum(Xf, observed, C, centers, use_abs, Iner_KD):
    N, D = Xf.shape
    K = centers.shape[0]

    # each k writes only Iner_KD[k, :]
    for k in prange(K):
        for i in range(N):
            c = C[i, k]
            if c == 0.0:
                continue
            for d in range(D):
                if not observed[i, d]:
                    continue
                diff = Xf[i, d] - centers[k, d]
                if use_abs:
                    Iner_KD[k, d] += c * abs(diff)
                else:
                    Iner_KD[k, d] += c * diff * diff
