"""Distribution families and parameter estimation for NEM."""

import numpy as np
from enum import Enum

# Three semantically distinct floors that used to share a single ``EPSILON``.
# They are all kept at 1e-20 so the numerics are byte-for-byte unchanged (the
# PPanGGOLiN reproduction and the Gaussian examples are preserved exactly); the
# split only makes the *intent* explicit and gives a single place to retune one
# without disturbing the others.
PROB_FLOOR = 1e-20      # floor for probabilities/proportions before a log
VAR_FLOOR = 1e-20       # floor for dispersions/variances (kept tiny to match
                        # the reference C core, where a degenerate dimension is
                        # handled by ZERO_DISP_TOL below, not by the floor)
ZERO_DISP_TOL = 1e-20   # a dispersion <= this marks a degenerate (point-mass)
                        # dimension: it contributes no density term, and an
                        # observation off the mode there gets zero likelihood
DIV_GUARD = 1e-20       # generic guard against division by zero (counts, sums)

# Back-compat alias: external code (and older imports) may still reference
# EPSILON. It maps to the generic division guard.
EPSILON = DIV_GUARD

# A class whose total (soft) membership falls below this is considered empty
# and reinitialised (k-means++ style) by _reinit_empty_classes.
EMPTY_CLASS_WEIGHT = 1.0


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


def compute_log_density(X, centers, dispersions, proportions, family,
                        weights=None):
    """Compute log(p_k * f_k(x_i)) for all i and k.

    Parameters
    ----------
    X : (N, D) array
    centers : (K, D) array
    dispersions : (K, D) array
    proportions : (K,) array
    family : Family enum
    weights : (D,) array or None
        Per-variable weights ``w_j > 0`` for the *weighted* NEM. Each variable's
        log-density contribution is multiplied by ``w_j`` before summing over D
        (a variable with ``w_j`` acts as if replicated ``w_j`` times). ``None``
        means all weights equal 1, i.e. the standard unweighted density —
        recovered byte-for-byte (the multiply is skipped entirely).

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
        log_pk = np.log(max(proportions[k], PROB_FLOOR))
        v = dispersions[k]                       # (D,)
        zero_v = v <= ZERO_DISP_TOL              # (D,)
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
        if weights is not None:
            contrib = contrib * weights[None, :]            # weighted NEM: w_j per variable
        log_fki = contrib.sum(axis=1)                       # (N,)

        # A zero-dispersion (point-mass) dimension gives -inf density to an
        # off-mode observation -- UNLESS that variable is weighted out (w_j = 0),
        # in which case it must be ignored entirely (the note's w_j=0 = variable
        # selection limit). weights=None -> all variables active -> exact old
        # behaviour, so the PPanGGOLiN reproduction is unaffected.
        active = (np.ones(D, dtype=bool) if weights is None
                  else weights > 0)
        invalid = (observed & zero_v[None, :] & (absdiff > ZERO_DISP_TOL)
                   & active[None, :]).any(axis=1)
        log_pkfki[:, k] = np.where(invalid, -np.inf, log_pk + log_fki)

    return log_pkfki


def estimate_parameters(X, C, family, dispersion_model, proportion_model,
                        miss_mode="replace", old_centers=None,
                        old_dispersions=None, weights=None):
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
    weights : (D,) array or None
        Per-variable weights for the weighted NEM. Centres and modes are
        **unchanged** by ``w_j`` (a per-column constant factors out of the
        weighted-likelihood normal equations, and out of the weighted median),
        so they are estimated as usual. The weights only matter for dispersion
        models that **pool across variables** (``s__`` and ``sk_``): there each
        column's inertia and count are weighted by ``w_j``. ``None`` = all ones.

    Returns
    -------
    dict with keys 'centers', 'dispersions', 'proportions'.
    """
    N, D = X.shape
    K = C.shape[1]
    observed = ~np.isnan(X)  # (N, D)

    # Class sizes
    raw_N_K = C.sum(axis=0)            # (K,) before clamping, for empty detection
    N_K = np.maximum(raw_N_K, DIV_GUARD)

    # Per-class, per-variable observed sizes
    N_KD = np.zeros((K, D))
    for k in range(K):
        N_KD[k] = (C[:, k:k+1] * observed).sum(axis=0)
    N_KD = np.maximum(N_KD, DIV_GUARD)

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
        Iner_KD, N_K, N_KD, N, D, K, dispersion_model, miss_mode, weights
    )

    # Clamp dispersions from below
    dispersions = np.maximum(dispersions, VAR_FLOOR)

    # --- Proportions ---
    if proportion_model == Proportion.EQUAL:
        proportions = np.full(K, 1.0 / K)
    else:
        proportions = N_K / N
        proportions = np.maximum(proportions, PROB_FLOOR)
        proportions /= proportions.sum()

    # --- Recover collapsed (empty) classes -------------------------------
    centers, dispersions, proportions = _reinit_empty_classes(
        X, centers, dispersions, proportions, raw_N_K, observed
    )

    return {
        "centers": centers,
        "dispersions": dispersions,
        "proportions": proportions,
    }


def _reinit_empty_classes(X, centers, dispersions, proportions, raw_N_K,
                          observed):
    """k-means++ style recovery for classes that have collapsed (emptied).

    A class whose total membership falls below ``EMPTY_CLASS_WEIGHT`` has its
    centre moved to the observation farthest from the dominant (largest) class
    centre — excluding points already chosen this round — its dispersion reset
    to the average dispersion of the populated classes, and its proportion
    floored so it can attract that point on the next E-step. The choice is
    deterministic (argmax distance, no RNG), so runs where no class is empty are
    left byte-for-byte unchanged (and PPanGGOLiN reproduction is preserved).
    """
    empty = np.where(raw_N_K < EMPTY_CLASS_WEIGHT)[0]
    if empty.size == 0:
        return centers, dispersions, proportions

    N = X.shape[0]
    Xf = np.where(observed, X, 0.0)
    populated = raw_N_K >= EMPTY_CLASS_WEIGHT
    if populated.any():
        default_disp = np.maximum(dispersions[populated].mean(axis=0), VAR_FLOOR)
    else:  # pathological: every class empty -> fall back to sample dispersion
        default_disp = np.maximum(np.nanvar(X, axis=0), VAR_FLOOR)
    dominant = int(np.argmax(raw_N_K))

    centers = centers.copy()
    dispersions = dispersions.copy()
    proportions = proportions.copy()

    # squared distance to the dominant centre, over observed variables
    d2 = ((Xf - centers[dominant]) ** 2 * observed).sum(axis=1)
    far_order = np.argsort(d2)[::-1]   # farthest observations first
    used = set()
    for k in empty:
        far = next((int(i) for i in far_order if i not in used),
                   int(far_order[0]))
        used.add(far)
        centers[k] = Xf[far]
        dispersions[k] = default_disp
        proportions[k] = max(proportions[k], 1.0 / N)

    proportions = proportions / proportions.sum()
    return centers, dispersions, proportions


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
        frac1 = np.where(W_total > 0, W_ones / np.maximum(W_total, DIV_GUARD), 0.0)
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


def _inertia_to_dispersions(Iner_KD, N_K, N_KD, N, D, K, model, miss_mode,
                            weights=None):
    """Convert inertia matrix to dispersions according to model.

    For the variable-pooling models (``s__``, ``sk_``) the sum over variables is
    weighted by ``weights`` (per-variable ``w_j``): the pooled variance maximises
    the weighted complete-data likelihood, giving ``sum_j w_j·Iner / sum_j w_j·N``
    (the column weight ``w_j`` cancels in the per-variable models ``skd``/``s_d``,
    which are therefore untouched). ``weights=None`` ⇒ all ones, recovering the
    original pooled estimators exactly (``sum_j w_j = D``).
    """
    dispersions = np.zeros((K, D))
    # Per-variable weights; w.sum() replaces the plain "D" column count in the
    # replace-mode denominators when pooling across variables.
    w = np.ones(D) if weights is None else np.asarray(weights, dtype=float)
    w_sum = w.sum()

    if model == Dispersion.S__:
        if miss_mode == "replace":
            v = (w * Iner_KD).sum() / (N * w_sum)
        else:
            v = (w * Iner_KD).sum() / (w * N_KD).sum()
        dispersions[:] = v

    elif model == Dispersion.SK_:
        for k in range(K):
            if miss_mode == "replace":
                vk = (w * Iner_KD[k]).sum() / (w_sum * N_K[k])
            else:
                vk = (w * Iner_KD[k]).sum() / (w * N_KD[k]).sum()
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
