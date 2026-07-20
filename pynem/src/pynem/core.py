"""Core NEM algorithm (E-step, M-step, convergence)."""

import networkx as nx
import numpy as np

from ._fast import HAS_NUMBA, seq_sweep
from .models import (
    DIV_GUARD,
    PROB_FLOOR,
    VAR_FLOOR,
    Dispersion,
    Family,
    Proportion,
    compute_log_density,
    estimate_parameters,
)
from .spatial import NeighborhoodSystem


class NEM:
    """Neighborhood EM for spatial clustering on graphs.

    Parameters
    ----------
    n_clusters : int
        Number of clusters K.
    beta : float
        Spatial regularization strength. 0 = standard EM.
    algorithm : str
        'nem' (mean field), 'ncem' (ICM hard), 'gem' (Gibbs sampling).
    family : str
        'normal', 'laplace', or 'bernoulli'.
    dispersion : str
        's__', 'sk_', 's_d', or 'skd'.
    proportion : str
        'p_' (equal) or 'pk' (free).
    beta_mode : str
        'fix', 'psgrad', 'heu_d', 'heu_l'.
    init : str
        'sort', 'random', or 'param' (initialize from given parameters,
        like NEM's INIT_PARAM_FILE — requires ``init_params``).
    init_params : tuple or None
        ``(centers, dispersions, proportions)`` used when ``init='param'``.
        ``centers`` and ``dispersions`` have shape (K, D), ``proportions`` (K,).
    site_update : str
        'parallel' (Jacobi: all nodes updated from the previous classification)
        or 'seq' (Gauss-Seidel: nodes updated in place, in index order, so a
        node sees the already-updated memberships of earlier neighbors). The
        reference NEM C code uses 'seq' (DEFAULT_UPDATE = UPDATE_SEQ); pynem
        keeps 'parallel' as default for backward compatibility.
    feature_weights : array-like of shape (D,) or None
        Per-variable weights ``w_j > 0`` for the *weighted* NEM (each feature/
        column contributes ``w_j`` times to the data fit). ``None`` (default)
        means all weights equal 1, i.e. the standard NEM — recovered exactly.
        Useful e.g. to down-weight redundant genomes in a pangenome so the
        partition reflects biology rather than sampling (see the weighted-NEM
        note). The M-step is unchanged by the weights except for the pooled
        dispersion models ``s__``/``sk_``.
    completeness : array-like of shape (D,) or None
        Per-genome completeness ``gamma_j in (0, 1]`` for the MAG-aware
        Bernoulli emission (mOTUpan-style): presence prob ``mu_kj * gamma_j``,
        so an absence in an incomplete genome is forgiven. Stops incomplete
        metagenome-assembled genomes from shrinking the persistent class.
        ``None`` (default) = full completeness, standard NEM. Bernoulli only.
    n_init : int
        Number of random initializations (only for init='random').
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold.
    convergence : str
        'classification' or 'criterion'.
    missing : str
        'replace' or 'ignore'.
    random_state : int or None
        Random seed.
    verbose : int
        Verbosity level.
    """

    def __init__(self, n_clusters=2, beta=1.0, algorithm="nem",
                 family="normal", dispersion="s__", proportion="pk",
                 beta_mode="fix", init="sort", init_params=None,
                 site_update="parallel", feature_weights=None,
                 completeness=None, n_init=1,
                 max_iter=100, tol=1e-3, convergence="classification",
                 missing="replace", random_state=None, verbose=0):
        self.n_clusters = n_clusters
        self.beta = beta
        self.algorithm = algorithm
        self.family = Family(family)
        self.dispersion = Dispersion(dispersion)
        self.proportion = Proportion(proportion)
        self.beta_mode = beta_mode
        self.init = init
        self.init_params = init_params
        self.site_update = site_update
        self.feature_weights = feature_weights
        self.completeness = completeness
        # Validated (D,) arrays, or None for the exact standard path. Set in
        # fit(); defaulted here so direct _initialize() calls also work.
        self._weights = None
        self._completeness = None
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.convergence = convergence
        self.missing = missing
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, G_or_X, graph=None):
        """Fit the NEM model.

        Parameters
        ----------
        G_or_X : nx.Graph or np.ndarray
            If nx.Graph: nodes must have 'features' attribute.
            If np.ndarray of shape (N, D): feature matrix.
        graph : nx.Graph or None
            Required if G_or_X is an array — provides the graph structure.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)

        if isinstance(G_or_X, nx.Graph):
            G = G_or_X
            self._check_graph(G)
            n = G.number_of_nodes()
            if "features" not in G.nodes[0]:
                raise ValueError("graph nodes must carry a 'features' attribute "
                                 "when fitting on a graph alone")
            d = len(G.nodes[0]["features"])
            X = np.array([G.nodes[i]["features"] for i in range(n)],
                         dtype=float)
        else:
            X = np.asarray(G_or_X, dtype=float)
            if X.ndim != 2:
                raise ValueError(f"X must be 2-D (N, D); got shape {X.shape}")
            G = graph
            if G is None:
                raise ValueError("a graph is required: pass fit(X, graph=...) "
                                 "or a single nx.Graph carrying node features")
            self._check_graph(G)
            n, d = X.shape
            if G.number_of_nodes() != n:
                raise ValueError(f"graph has {G.number_of_nodes()} nodes but X "
                                 f"has {n} rows; they must match")

        K = self.n_clusters
        if d < 1:
            raise ValueError("X must have at least one feature column (D >= 1)")
        if not isinstance(K, (int, np.integer)) or K < 1:
            raise ValueError(f"n_clusters must be an integer >= 1; got {K}")
        if K > n:
            raise ValueError(f"n_clusters={K} exceeds the number of nodes "
                             f"N={n}; need K <= N")
        if self.init == "param" and self.init_params is None:
            raise ValueError("init='param' requires init_params=(centers, "
                             "dispersions, proportions)")

        # Per-variable weights for the weighted NEM. ``None`` keeps the exact
        # unweighted path (no array multiply); otherwise validate shape > 0.
        if self.feature_weights is None:
            self._weights = None
        else:
            w = np.asarray(self.feature_weights, dtype=float)
            if w.shape != (d,):
                raise ValueError(f"feature_weights must have shape ({d},) to "
                                 f"match the {d} features; got {w.shape}")
            if np.any(w < 0) or not np.all(np.isfinite(w)):
                raise ValueError("feature_weights must be finite and "
                                 "non-negative")
            self._weights = w

        # Per-genome completeness for the MAG-aware Bernoulli model. ``None``
        # keeps the standard path; otherwise validate shape and range (0, 1].
        if self.completeness is None:
            self._completeness = None
        else:
            if self.family != Family.BERNOULLI:
                raise ValueError("completeness is only supported for "
                                 "family='bernoulli'")
            gamma = np.asarray(self.completeness, dtype=float)
            if gamma.shape != (d,):
                raise ValueError(f"completeness must have shape ({d},) to match "
                                 f"the {d} features/genomes; got {gamma.shape}")
            if np.any(gamma <= 0) or np.any(gamma > 1) or not np.all(np.isfinite(gamma)):
                raise ValueError("completeness must lie in (0, 1]")
            self._completeness = gamma

        ns = NeighborhoodSystem(G)

        if self.beta_mode in ("heu_d", "heu_l"):
            self._fit_heuristic(X, G, ns, K, rng)
            return self

        n_runs = self.n_init if self.init == "random" else 1
        best_crit = -np.inf
        best_result = None

        for run in range(n_runs):
            result = self._run_once(X, ns, K, rng)
            crit_val = result["criteria"]["U"]
            # Always keep the only/first run, even if its criterion is not
            # finite (NaN > -inf is False), so a single fit never returns None.
            if best_result is None or crit_val > best_crit:
                best_crit = crit_val
                best_result = result

        self._store_result(best_result)
        return self

    @staticmethod
    def _check_graph(G):
        """Validate that the graph is usable: nodes labelled 0..n-1.

        pynem indexes nodes positionally (row i of X <-> node i, CSR row i),
        so the node set must be exactly {0, ..., n-1}. networkx graphs with
        string labels, gaps, or 1-based labels are rejected with a clear
        message rather than failing obscurely deeper in the pipeline.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError(f"expected an nx.Graph (or DiGraph); got {type(G)}")
        n = G.number_of_nodes()
        if n == 0:
            raise ValueError("graph has no nodes")
        nodes = set(G.nodes)
        if nodes != set(range(n)):
            raise ValueError("graph nodes must be the contiguous integers "
                             f"0..{n - 1}; relabel with "
                             "networkx.convert_node_labels_to_integers")

    # Constructor parameters, for the scikit-learn-style get_params/set_params.
    _PARAM_NAMES = (
        "n_clusters", "beta", "algorithm", "family", "dispersion", "proportion",
        "beta_mode", "init", "init_params", "site_update", "feature_weights",
        "completeness", "n_init", "max_iter", "tol", "convergence", "missing",
        "random_state", "verbose",
    )

    def get_params(self, deep=True):
        """Get parameters for this estimator (scikit-learn compatible).

        Returns the constructor arguments as a dict, so ``NEM(**est.get_params())``
        reconstructs an equivalent estimator (enum-valued params are returned as
        their string form, e.g. ``"normal"``).
        """
        params = {}
        for name in self._PARAM_NAMES:
            value = getattr(self, name)
            if isinstance(value, (Family, Dispersion, Proportion)):
                value = value.value
            params[name] = value
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator (scikit-learn compatible).

        Enables use in tools that call ``set_params`` (grid search, clone). The
        enum-valued parameters accept their string form.
        """
        enum_map = {"family": Family, "dispersion": Dispersion,
                    "proportion": Proportion}
        for name, value in params.items():
            if name not in self._PARAM_NAMES:
                raise ValueError(
                    f"invalid parameter {name!r} for estimator NEM; valid "
                    f"parameters are {sorted(self._PARAM_NAMES)}")
            if name in enum_map:
                value = enum_map[name](value)
            setattr(self, name, value)
        return self

    def _check_fitted(self):
        if not hasattr(self, "labels_"):
            raise RuntimeError("This NEM instance is not fitted yet; call "
                               "fit() before using this method.")

    def _posterior(self, X):
        """Soft membership of new feature vectors under the fitted model.

        Uses the fitted parameters and the data likelihood only (no spatial
        term — new points are not part of the training graph). Shape (N, K).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.centers_.shape[1]:
            raise ValueError(
                f"X must be 2-D with {self.centers_.shape[1]} features; "
                f"got shape {X.shape}")
        log_pkfki = compute_log_density(
            X, self.centers_, self.dispersions_, self.proportions_,
            self.family, weights=self._weights, completeness=self._completeness,
        )
        return self._normalize_membership(log_pkfki, np.zeros_like(log_pkfki))

    def predict(self, X=None):
        """Hard labels (1-based).

        ``predict()`` returns the fitted-data labels. ``predict(X)`` classifies
        new feature vectors ``X`` (N, D) by the fitted mixture (spatial term
        excluded, since new points are not in the training graph).
        """
        if X is None:
            self._check_fitted()
            return self.labels_
        return np.argmax(self._posterior(X), axis=1) + 1

    def transform(self, X=None):
        """Soft membership (N, K).

        ``transform()`` returns the fitted-data membership; ``transform(X)`` the
        membership of new feature vectors ``X``.
        """
        if X is None:
            self._check_fitted()
            return self.membership_
        return self._posterior(X)

    def fit_predict(self, G_or_X, graph=None):
        """Fit the model and return the hard labels of the training data."""
        self.fit(G_or_X, graph=graph)
        return self.labels_

    def score(self, X=None, y=None):
        """Mixture log-likelihood under the fitted model (higher is better).

        ``score()`` returns the fitted-data log-likelihood (``criteria_['L']``);
        ``score(X)`` evaluates it on new feature vectors ``X``. ``y`` is ignored
        (present for scikit-learn API compatibility).
        """
        self._check_fitted()
        if X is None:
            return float(self.criteria_["L"])
        from scipy.special import logsumexp
        log_pkfki = compute_log_density(
            np.asarray(X, dtype=float), self.centers_, self.dispersions_,
            self.proportions_, self.family, weights=self._weights,
            completeness=self._completeness,
        )
        return float(logsumexp(log_pkfki, axis=1).sum())

    def _run_once(self, X, ns, K, rng):
        """Single NEM run from one initialization."""
        N, D = X.shape
        beta = self.beta

        # X doesn't change across this run's iterations, so its NaN mask and
        # NaN-filled copy are computed once here and threaded through every
        # compute_log_density/estimate_parameters call below, instead of each
        # call reallocating its own full (N, D) array every iteration

        observed = ~np.isnan(X)
        Xf = np.where(observed, X, 0.0)

        # Initialize
        C = self._initialize(X, ns, K, rng, observed=observed, Xf=Xf)

        # Compute sample statistics for random init
        history = []
        old_C = None
        old_crit = None

        params = None
        for iteration in range(self.max_iter):
            # M-step (the first one has no previous parameters to reuse)
            if iteration == 0:
                params = self._first_m_step(X, C, observed=observed, Xf=Xf)
            else:
                params = estimate_parameters(
                    X, C, self.family, self.dispersion, self.proportion,
                    miss_mode=self.missing,
                    old_centers=params["centers"],
                    old_dispersions=params["dispersions"],
                    weights=self._weights, completeness=self._completeness,
                    observed=observed, Xf=Xf,
                )

            # Beta estimation (pseudo-gradient)
            if self.beta_mode == "psgrad":
                beta = self._estimate_beta(C, ns, beta)

            # log(p_k f_k(x_i)) depends only on the current parameters, so it is
            # identical in the E-step and in the criteria below — compute it once.
            log_pkfki = compute_log_density(
                X, params["centers"], params["dispersions"],
                params["proportions"], self.family,
                weights=self._weights, completeness=self._completeness,
                observed=observed, Xf=Xf,
            )

            # E-step
            old_C = C.copy()
            C = self._e_step(X, C, params, ns, beta, K, rng,
                             log_pkfki=log_pkfki, observed=observed, Xf=Xf)

            # Compute criteria
            criteria = self._compute_criteria(X, C, params, ns, beta,
                                              log_pkfki=log_pkfki)
            history.append(criteria.copy())

            if self.verbose:
                print(f"  iter {iteration}: U={criteria['U']:.4f} "
                      f"D={criteria['D']:.4f} G={criteria['G']:.4f} "
                      f"beta={beta:.4f}")

            # Convergence test
            if old_crit is not None and self._has_converged(old_C, C,
                                                            old_crit, criteria):
                break
            old_crit = criteria

        return {
            "membership": C,
            "labels": np.argmax(C, axis=1) + 1,
            "centers": params["centers"],
            "dispersions": params["dispersions"],
            "proportions": params["proportions"],
            "beta": beta,
            "criteria": criteria,
            "n_iter": iteration + 1,
            "history": history,
        }

    def _first_m_step(self, X, C, observed=None, Xf=None):
        """First M-step (no old parameters)."""
        return estimate_parameters(
            X, C, self.family, self.dispersion, self.proportion,
            miss_mode=self.missing, weights=self._weights, completeness=self._completeness,
            observed=observed, Xf=Xf,
        )

    def _initialize(self, X, ns, K, rng, observed=None, Xf=None):
        """Initialize the classification matrix C (N, K)."""
        N, D = X.shape
        C = np.zeros((N, K))

        if self.init == "param":
            # Initialize from given parameters, mirroring NEM's
            # ComputePartitionFromPara(Needinit=1): a "blind" E-step with
            # beta=0 (pure mixture posterior), then one spatial E-step at the
            # actual beta.
            if self.init_params is None:
                raise ValueError("init='param' requires init_params=("
                                 "centers, dispersions, proportions)")
            centers, dispersions, proportions = self.init_params
            params = {
                "centers": np.asarray(centers, dtype=float),
                "dispersions": np.maximum(np.asarray(dispersions, dtype=float),
                                          VAR_FLOOR),
                "proportions": np.asarray(proportions, dtype=float),
            }
            log_pkfki = compute_log_density(
                X, params["centers"], params["dispersions"],
                params["proportions"], self.family,
                weights=self._weights, completeness=self._completeness,
                observed=observed, Xf=Xf,
            )
            C_blind = self._normalize_membership(log_pkfki, np.zeros((N, K)))
            C = self._e_step(X, C_blind, params, ns, self.beta, K, rng,
                             observed=observed, Xf=Xf)

        elif self.init == "sort":
            # Sort by first variable, partition into K equal groups
            sorted_var = 0
            vals = X[:, sorted_var].copy()
            # Handle NaN by setting to max for sorting
            nan_mask = np.isnan(vals)
            vals[nan_mask] = np.nanmax(vals) + 1 if not nan_mask.all() else 0
            order = np.argsort(vals)
            for rank, idx in enumerate(order):
                ki = (rank * K) // N
                ki = min(ki, K - 1)
                C[idx, ki] = 1.0

        elif self.init == "random":
            # Pick K distinct data points as centers, then do one E-step
            if observed is None:
                observed = ~np.isnan(X)
            sample_disp = np.nanvar(X, axis=0)
            sample_disp = np.maximum(sample_disp, VAR_FLOOR)

            # Pick K distinct centers
            centers = np.zeros((K, D))
            chosen = []
            for k in range(K):
                for _ in range(100):
                    idx = rng.integers(0, N)
                    if idx not in chosen:
                        chosen.append(idx)
                        break
                for d in range(D):
                    if observed[idx, d]:
                        centers[k, d] = X[idx, d]
                    else:
                        lo = np.nanmin(X[:, d]) if observed[:, d].any() else 0
                        hi = np.nanmax(X[:, d]) if observed[:, d].any() else 1
                        centers[k, d] = rng.uniform(lo, hi)

            dispersions = np.tile(sample_disp / K, (K, 1))
            proportions = np.full(K, 1.0 / K)

            # One E-step to get initial classification
            log_pkfki = compute_log_density(
                X, centers, dispersions, proportions, self.family,
                weights=self._weights, completeness=self._completeness,
                observed=observed, Xf=Xf,
            )
            # Without spatial term (beta=0 for init)
            C = self._normalize_membership(log_pkfki, np.zeros((N, K)))

        return C

    def _normalize_local(self, log_num, K):
        """Normalize one node's log-numerators into a membership vector."""
        max_log = np.max(log_num)
        if np.isfinite(max_log):
            num = np.exp(log_num - max_log)
            total = num.sum()
            if total > 0:
                return num / total
        return np.full(K, 1.0 / K)

    def _harden_node(self, c_i, K, rng):
        """Apply the C-step (ncem/gem) to a single node's membership."""
        if self.algorithm == "ncem":
            hard = np.zeros(K)
            hard[np.argmax(c_i)] = 1.0
            return hard
        elif self.algorithm == "gem":
            hard = np.zeros(K)
            hard[rng.choice(K, p=c_i)] = 1.0
            return hard
        return c_i

    def _e_step(self, X, C, params, ns, beta, K, rng, log_pkfki=None,
               observed=None, Xf=None):
        """E-step: update classification (one sweep).

        With ``site_update='seq'`` the update is Gauss-Seidel (in place, index
        order) — the reference NEM behaviour. With ``'parallel'`` it is Jacobi
        (every node uses the classification from the start of the sweep).

        ``log_pkfki`` (the (N, K) log(p_k f_k) for the current params) may be
        supplied to avoid recomputing it when the caller already has it.
        """
        N = X.shape[0]

        if log_pkfki is None:
            log_pkfki = compute_log_density(
                X, params["centers"], params["dispersions"],
                params["proportions"], self.family,
                weights=self._weights, completeness=self._completeness,
                observed=observed, Xf=Xf,
            )

        if self.site_update == "seq":
            # Gauss-Seidel: update CM in place, visiting nodes 0..N-1.
            CM = np.array(C, dtype=np.float64, order="C")
            if HAS_NUMBA and self.algorithm in ("nem", "ncem"):
                indptr, indices, data = ns.csr_arrays()
                return seq_sweep(
                    CM, np.ascontiguousarray(log_pkfki, dtype=np.float64),
                    indptr, indices, data,
                    float(beta), self.algorithm == "ncem",
                )
            # Pure-Python fallback (used without numba, and for Gibbs/gem).
            for i in range(N):
                context = ns.spatial_context(i, CM, K)
                c_i = self._normalize_local(log_pkfki[i] + beta * context, K)
                CM[i] = self._harden_node(c_i, K, rng)
            return CM

        # Parallel (Jacobi): all nodes from the previous classification,
        # vectorised — context = A @ C, then a single log-sum-exp normalisation.
        contexts = ns.compute_all_contexts(C)
        new_C = self._normalize_membership(log_pkfki, beta * contexts)

        if self.algorithm == "ncem":
            hard = np.zeros_like(new_C)
            hard[np.arange(N), np.argmax(new_C, axis=1)] = 1.0
            new_C = hard
        elif self.algorithm == "gem":
            hard = np.zeros_like(new_C)
            for i in range(N):
                k = rng.choice(K, p=new_C[i])
                hard[i, k] = 1.0
            new_C = hard

        return new_C

    def _normalize_membership(self, log_pkfki, spatial_term):
        """Normalize log-probabilities to membership probabilities."""
        N, K = log_pkfki.shape
        log_num = log_pkfki + spatial_term
        # Log-sum-exp normalization
        max_log = np.max(log_num, axis=1, keepdims=True)
        finite = np.isfinite(max_log.ravel())
        C = np.full((N, K), 1.0 / K)
        if finite.any():
            num = np.exp(log_num[finite] - max_log[finite])
            total = num.sum(axis=1, keepdims=True)
            total = np.maximum(total, DIV_GUARD)
            C[finite] = num / total
        return C

    def _estimate_beta(self, C, ns, beta, max_beta_iter=50, step=0.0):
        """Pseudo-gradient ascent on pseudo-likelihood for beta."""
        N, K = C.shape
        contexts = ns.compute_all_contexts(C)

        for _ in range(max_beta_iter):
            # Gradient and second derivative of pseudo-likelihood
            # Q(beta) = sum_i [beta * sum_k c_ik * con_ik - log(sum_k exp(beta*con_ik))]
            beta_con = beta * contexts  # (N, K)
            max_bc = np.max(beta_con, axis=1, keepdims=True)
            exp_bc = np.exp(beta_con - max_bc)
            Z = exp_bc.sum(axis=1, keepdims=True)  # (N, 1)

            # sum_k c_ik * con_ik
            c_dot_con = (C * contexts).sum(axis=1)  # (N,)

            # sum_k con_ik * exp(beta*con_ik) / Z_i
            mean_con = (contexts * exp_bc / Z).sum(axis=1)  # (N,)

            grad = (c_dot_con - mean_con).sum()

            # Second derivative
            mean_con2 = (contexts ** 2 * exp_bc / Z).sum(axis=1)
            d2 = (mean_con2 - mean_con ** 2).sum()

            if abs(grad) < self.tol * N:
                break

            if step <= 0 and abs(d2) > DIV_GUARD:
                # Newton step with damping factor 4
                beta += grad / (4 * abs(d2))
            else:
                beta += grad * (step / N) if step > 0 else 0.01

            beta = np.clip(beta, -5.0, 5.0)

        return beta

    def _compute_criteria(self, X, C, params, ns, beta, log_pkfki=None):
        """Compute all criteria: U, D, G, L, M.

        ``log_pkfki`` may be supplied (it depends only on ``params``) to reuse
        the array already computed by the E-step in the same iteration.
        """
        N, K = C.shape
        if log_pkfki is None:
            log_pkfki = compute_log_density(
                X, params["centers"], params["dispersions"],
                params["proportions"], self.family,
                weights=self._weights, completeness=self._completeness,
            )

        # D (Hathaway) = sum_i sum_k c_ik * (log(pk*fki) - log(c_ik)), summed
        # only where c_ik > 0 (0·log0 = 0 by convention). This also avoids
        # 0·(-inf) = NaN when a class has zero dispersion on some variable
        # (e.g. a gene family present in every genome -> mode 1, dispersion 0),
        # which gives -inf density to non-members that are not assigned to it.
        log_C = np.log(np.maximum(C, PROB_FLOOR))
        mask = C > 0
        D = (C[mask] * (log_pkfki[mask] - log_C[mask])).sum()

        # G (geographic cohesion)
        contexts = ns.compute_all_contexts(C)
        G = (C * contexts).sum()

        # U (NEM criterion)
        U = D + 0.5 * beta * G

        # L (mixture log-likelihood)
        max_log = np.max(log_pkfki, axis=1, keepdims=True)
        finite = np.isfinite(max_log.ravel())
        L = 0.0
        if finite.any():
            sum_exp = np.sum(np.exp(log_pkfki[finite] - max_log[finite]),
                             axis=1)
            L = (max_log[finite].ravel() + np.log(np.maximum(sum_exp, PROB_FLOOR))).sum()

        # Z and M (Markovian pseudo-likelihood)
        beta_con = beta * contexts
        max_bc = np.max(beta_con, axis=1)
        Z_vals = max_bc + np.log(np.sum(np.exp(beta_con - max_bc[:, None]),
                                        axis=1))
        Z = -Z_vals.sum()
        M = D + beta * G + Z

        return {"U": U, "D": D, "G": G, "L": L, "M": M}

    def _has_converged(self, C_old, C_new, crit_old, crit_new):
        """Check convergence."""
        if self.convergence == "classification":
            return np.max(np.abs(C_new - C_old)) < self.tol
        elif self.convergence == "criterion":
            if abs(crit_new["U"]) < DIV_GUARD:
                return True
            return abs(crit_new["U"] - crit_old["U"]) / abs(crit_new["U"]) < self.tol
        return False

    def _fit_heuristic(self, X, G, ns, K, rng):
        """Heuristic beta estimation: increase beta until criterion drops."""
        # First run with beta=0 (pure EM)
        self.beta = 0.0
        result0 = self._run_once(X, ns, K, rng)
        D0 = result0["criteria"]["D"]
        L0 = result0["criteria"]["L"]

        best_result = result0
        best_U = result0["criteria"]["U"]

        beta_step = 0.1
        max_beta = 2.0

        for trial_beta in np.arange(beta_step, max_beta + beta_step, beta_step):
            self.beta = trial_beta
            result = self._run_once(X, ns, K, rng)

            # Check stopping criterion
            if self.beta_mode == "heu_d":
                if result["criteria"]["D"] < 0.8 * D0:
                    break
            elif self.beta_mode == "heu_l":
                if abs(result["criteria"]["L"]) < 0.02 * abs(L0):
                    break

            if result["criteria"]["U"] > best_U:
                best_U = result["criteria"]["U"]
                best_result = result

        self._store_result(best_result)

    def _store_result(self, result):
        """Store algorithm results as attributes."""
        self.labels_ = result["labels"]
        self.membership_ = result["membership"]
        self.centers_ = result["centers"]
        self.dispersions_ = result["dispersions"]
        self.proportions_ = result["proportions"]
        self.beta_ = result["beta"]
        self.criteria_ = result["criteria"]
        self.n_iter_ = result["n_iter"]
        self.history_ = result["history"]
