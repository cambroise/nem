"""Distribution families and parameter estimation for NEM."""

import numpy as np
from enum import Enum

EPSILON = 1e-20


class Family(Enum):
    NORMAL = "normal"
    LAPLACE = "laplace"
    BERNOULLI = "bernoulli"


class Dispersion(Enum):
    S__ = "s__"   # single dispersion for all
    SK_ = "sk_"   # per-class
    S_D = "s_d"   # per-variable
    SKD = "skd"   # full (per-class, per-variable)


class Proportion(Enum):
    EQUAL = "p_"  # equal proportions = 1/K
    FREE = "pk"   # free proportions


def compute_log_density(X, centers, dispersions, proportions, family):
    """Compute log(p_k * f_k(x_i)) for all i and k.

    Parameters
    ----------
    X : (N, D) array
    centers : (K, D) array
    dispersions : (K, D) array
    proportions : (K,) array
    family : Family enum

    Returns
    -------
    log_pkfki : (N, K) array
        log(p_k * f_k(x_i)) for each observation i and class k.
    """
    N, D = X.shape
    K = centers.shape[0]
    log_pkfki = np.empty((N, K))

    observed = ~np.isnan(X)            # (N, D)
    Xf = np.where(observed, X, 0.0)    # NaN -> 0 (these terms are masked out)

    # Vectorised over D; the K loop is cheap (K is small). For each class we
    # build the per-(observation, variable) log-density contributions, zero out
    # unobserved and zero-dispersion dimensions, then sum over D. A density is
    # exactly zero (log = -inf) when an observed variable has zero dispersion
    # yet the observation differs from the centre.
    for k in range(K):
        log_pk = np.log(max(proportions[k], EPSILON))
        v = dispersions[k]                       # (D,)
        zero_v = v <= EPSILON                    # (D,)
        diff = Xf - centers[k]                   # (N, D)
        absdiff = np.abs(diff)

        if family == Family.NORMAL:
            # -0.5 [ log(2π v) + (x - m)² / v ]
            safe_v = np.where(zero_v, 1.0, v)
            contrib = -0.5 * (np.log(2 * np.pi * safe_v) + diff * diff / safe_v)
        elif family == Family.LAPLACE:
            # -[ log(2 v) + |x - m| / v ]
            safe_v = np.where(zero_v, 1.0, v)
            contrib = -(np.log(2 * safe_v) + absdiff / safe_v)
        else:  # BERNOULLI: -[ -log(1 - v) + |x - m| log((1 - v) / v) ]
            safe_v = np.where(zero_v, 0.5, v)
            contrib = np.log(1 - safe_v) - absdiff * np.log((1 - safe_v) / safe_v)

        contrib = np.where(zero_v[None, :], 0.0, contrib)   # zero-disp dims: no term
        contrib = np.where(observed, contrib, 0.0)          # unobserved dims: no term
        log_fki = contrib.sum(axis=1)                       # (N,)

        invalid = (observed & zero_v[None, :] & (absdiff > EPSILON)).any(axis=1)
        log_pkfki[:, k] = np.where(invalid, -np.inf, log_pk + log_fki)

    return log_pkfki


def estimate_parameters(X, C, family, dispersion_model, proportion_model,
                        miss_mode="replace", old_centers=None,
                        old_dispersions=None):
    """M-step: estimate model parameters from soft classification.

    Parameters
    ----------
    X : (N, D) array
    C : (N, K) array — soft classification
    family : Family enum
    dispersion_model : Dispersion enum
    proportion_model : Proportion enum
    miss_mode : str — 'replace' or 'ignore'
    old_centers : (K, D) array or None
    old_dispersions : (K, D) array or None

    Returns
    -------
    dict with keys 'centers', 'dispersions', 'proportions'.
    """
    N, D = X.shape
    K = C.shape[1]
    observed = ~np.isnan(X)  # (N, D)

    # Class sizes
    N_K = C.sum(axis=0)  # (K,)
    N_K = np.maximum(N_K, EPSILON)

    # Per-class, per-variable observed sizes
    N_KD = np.zeros((K, D))
    for k in range(K):
        N_KD[k] = (C[:, k:k+1] * observed).sum(axis=0)
    N_KD = np.maximum(N_KD, EPSILON)

    # --- Centers ---
    # Bernoulli, like Laplace, uses the weighted MEDIAN as center estimator
    # (the reference NEM C code routes FAMILY_BERNOULLI to EstimParaLaplace).
    # For 0/1 data the weighted median is the binary MODE, which equals
    # thresholding the C-weighted fraction of 1s at 0.5 — computed here as two
    # matmuls, with no per-(k,d) sort (see _estimate_bernoulli_centers).
    if family == Family.BERNOULLI:
        centers = _estimate_bernoulli_centers(X, C, observed)
    elif family == Family.LAPLACE:
        centers = _estimate_laplace_centers(X, C, observed, K, D, N_K)
    else:
        centers = _estimate_mean_centers(X, C, observed, K, D, N, N_K, N_KD,
                                         miss_mode, old_centers)

    # --- Inertia ---
    Iner_KD = np.zeros((K, D))
    X_filled = X.copy()
    X_filled[~observed] = 0.0

    for k in range(K):
        diff = X_filled - centers[k]
        diff[~observed] = 0.0
        if family == Family.NORMAL:
            Iner_KD[k] = (C[:, k:k+1] * diff ** 2 * observed).sum(axis=0)
        else:  # Laplace or Bernoulli
            Iner_KD[k] = (C[:, k:k+1] * np.abs(diff) * observed).sum(axis=0)

        # Missing data correction for REPLACE mode (Normal only)
        if miss_mode == "replace" and family == Family.NORMAL and old_dispersions is not None:
            n_miss_kd = N_K[k] - N_KD[k]
            Iner_KD[k] += np.maximum(n_miss_kd, 0) * old_dispersions[k]

    # --- Dispersions ---
    dispersions = _inertia_to_dispersions(
        Iner_KD, N_K, N_KD, N, D, K, dispersion_model, miss_mode
    )

    # Clamp dispersions from below
    dispersions = np.maximum(dispersions, EPSILON)

    # --- Proportions ---
    if proportion_model == Proportion.EQUAL:
        proportions = np.full(K, 1.0 / K)
    else:
        proportions = N_K / N
        proportions = np.maximum(proportions, EPSILON)
        proportions /= proportions.sum()

    return {
        "centers": centers,
        "dispersions": dispersions,
        "proportions": proportions,
    }


def _estimate_mean_centers(X, C, observed, K, D, N, N_K, N_KD, miss_mode,
                           old_centers):
    """Weighted mean center estimation (Normal, Bernoulli)."""
    centers = np.zeros((K, D))
    X_filled = X.copy()
    X_filled[~observed] = 0.0

    for k in range(K):
        weighted_sum = (C[:, k:k+1] * X_filled * observed).sum(axis=0)
        if miss_mode == "replace" and old_centers is not None:
            n_miss_kd = N_K[k] - N_KD[k]
            weighted_sum += np.maximum(n_miss_kd, 0) * old_centers[k]
            centers[k] = weighted_sum / N_K[k]
        else:
            centers[k] = weighted_sum / N_KD[k]

    return centers


def _estimate_bernoulli_centers(X, C, observed):
    """Bernoulli center = binary mode (weighted median of {0,1}), vectorised.

    For 0/1 data the C-weighted median equals 1 iff the weighted fraction of 1s
    exceeds 0.5, which is exactly what the weighted median returns (ties at 0.5
    map to 0, as in ``_estimate_laplace_centers``). Computed with two matmuls
    instead of a per-(class, variable) sort.

    Returns
    -------
    centers : (K, D) array of {0.0, 1.0}
    """
    Xf = np.where(observed, X, 0.0)
    obs = observed.astype(float)
    W_total = C.T @ obs        # (K, D) sum of weights over observed entries
    W_ones = C.T @ Xf          # (K, D) weighted count of 1s
    with np.errstate(invalid="ignore", divide="ignore"):
        frac1 = np.where(W_total > 0, W_ones / np.maximum(W_total, EPSILON), 0.0)
    return (frac1 > 0.5).astype(float)


def _estimate_laplace_centers(X, C, observed, K, D, N_K):
    """Weighted median center estimation for Laplace family."""
    centers = np.zeros((K, D))
    for k in range(K):
        weights = C[:, k]
        for d in range(D):
            obs_mask = observed[:, d]
            if obs_mask.sum() == 0:
                centers[k, d] = 0.0
                continue
            vals = X[obs_mask, d]
            w = weights[obs_mask]
            # Weighted median
            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            w_sorted = w[idx]
            cumw = np.cumsum(w_sorted)
            half = cumw[-1] / 2.0
            median_idx = np.searchsorted(cumw, half)
            median_idx = min(median_idx, len(vals_sorted) - 1)
            centers[k, d] = vals_sorted[median_idx]
    return centers


def _inertia_to_dispersions(Iner_KD, N_K, N_KD, N, D, K, model, miss_mode):
    """Convert inertia matrix to dispersions according to model."""
    dispersions = np.zeros((K, D))

    if model == Dispersion.S__:
        if miss_mode == "replace":
            v = Iner_KD.sum() / (N * D)
        else:
            v = Iner_KD.sum() / N_KD.sum()
        dispersions[:] = v

    elif model == Dispersion.SK_:
        for k in range(K):
            if miss_mode == "replace":
                vk = Iner_KD[k].sum() / (D * N_K[k])
            else:
                vk = Iner_KD[k].sum() / N_KD[k].sum()
            dispersions[k, :] = vk

    elif model == Dispersion.S_D:
        for d in range(D):
            if miss_mode == "replace":
                vd = Iner_KD[:, d].sum() / N
            else:
                vd = Iner_KD[:, d].sum() / N_KD[:, d].sum()
            dispersions[:, d] = vd

    elif model == Dispersion.SKD:
        if miss_mode == "replace":
            for k in range(K):
                dispersions[k] = Iner_KD[k] / N_K[k]
        else:
            dispersions = Iner_KD / N_KD

    return dispersions
