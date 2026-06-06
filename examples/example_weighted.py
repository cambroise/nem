"""Example: weighted NEM — correcting genome redundancy with automatic weights.

Some genomes are redundant (an over-sampled clade): they carry the same signal
many times and bias a mixture/NEM partition toward their own biology. pynem can
down-weight them automatically:

  1. ``pynem.genome_weights`` groups the genomes (columns) by Jaccard distance
     and UPGMA (hierarchical clustering), choosing the number of groups by the
     silhouette criterion, and turns the groups into inverse-abundance weights;
  2. those weights are passed to ``NEM(feature_weights=...)`` — only the E-step
     changes, the closed-form M-step is preserved.

Here 6 "signal" genomes carry the TRUE family partition t, while 60 redundant
genomes carry an unrelated SPURIOUS partition s. Unweighted, the 60 redundant
columns dominate; with automatic weights, the true partition is recovered.

Uses only pynem (no PPanGGOLiN install required). Saves a figure for the README.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pynem" / "src"))

from pynem import NEM, genome_weights
from pynem.weights import jaccard_upgma_linkage, jaccard_upgma_labels, \
    silhouette_score
from pynem.metrics import adjusted_rand_index as ari

HERE = Path(__file__).parent

# ── Simulate redundancy ──────────────────────────────────────────────────────
rng = np.random.default_rng(0)
n = 400                                   # gene families (rows)
t = rng.integers(0, 2, n)                 # TRUE partition (signal)
s = rng.integers(0, 2, n)                 # SPURIOUS partition (confounder)


def block(label, ncol, p1=0.9, p0=0.1):
    P = np.where(label[:, None] == 1, p1, p0)
    return (rng.random((n, ncol)) < P).astype(float)


N_SIGNAL, N_REDUNDANT = 6, 60
X = np.hstack([block(t, N_SIGNAL), block(s, N_REDUNDANT)])   # (n, 66) genomes
G = nx.cycle_graph(n)                       # inert graph (beta = 0 isolates weights)

# ── Automatic weights: Jaccard + UPGMA on the genomes (columns) ──────────────
weights, groups = genome_weights(X, selection="silhouette", k_range=range(2, 8),
                                 return_labels=True)
n_groups = len(np.unique(groups))
print(f"Jaccard+UPGMA found {n_groups} genome groups (silhouette); "
      f"sizes {sorted(np.bincount(groups)[1:], reverse=True)}")
print(f"weights: signal genomes ~{weights[:N_SIGNAL].mean():.2f}, "
      f"redundant genomes ~{weights[N_SIGNAL:].mean():.2f}")


def labels(w):
    m = NEM(n_clusters=2, beta=0.0, family="bernoulli", dispersion="skd",
            proportion="pk", init="random", n_init=5, site_update="seq",
            feature_weights=w, random_state=1, max_iter=100)
    m.fit(X, graph=G)
    return m.labels_


lab_u, lab_w = labels(None), labels(weights)
ari_u_t, ari_u_s = ari(t, lab_u), ari(s, lab_u)
ari_w_t, ari_w_s = ari(t, lab_w), ari(s, lab_w)
print(f"unweighted : ARI(true t) = {ari_u_t:+.3f}   ARI(spurious s) = {ari_u_s:+.3f}")
print(f"weighted   : ARI(true t) = {ari_w_t:+.3f}   ARI(spurious s) = {ari_w_s:+.3f}")

# ── Figure ───────────────────────────────────────────────────────────────────
condensed, Z = jaccard_upgma_linkage(X.T)          # genomes = columns
Dsq = squareform(condensed)
ks = list(range(2, 8))
sils = [silhouette_score(Dsq, fcluster(Z, t=k, criterion="maxclust")) for k in ks]
best_k = ks[int(np.argmax(sils))]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) dendrogram of the genomes, cut into n_groups
ax = axes[0, 0]
# colour threshold just below the (n_groups-1)-th largest merge height
heights = np.sort(Z[:, 2])
ct = heights[-(n_groups - 1)] - 1e-9 if n_groups > 1 else 0
dendrogram(Z, ax=ax, color_threshold=ct, no_labels=True)
ax.axhline(ct, color="grey", ls="--", lw=1)
ax.set_title(f"(a) Jaccard + UPGMA dendrogram of the {X.shape[1]} genomes\n"
             f"silhouette cut → {n_groups} groups")
ax.set_ylabel("Jaccard distance")

# (b) silhouette vs number of groups
ax = axes[0, 1]
ax.plot(ks, sils, "o-")
ax.axvline(best_k, color="C3", ls="--", lw=1, label=f"chosen k = {best_k}")
ax.set_xlabel("number of groups k"); ax.set_ylabel("mean silhouette")
ax.set_title("(b) Choosing k by silhouette"); ax.legend()

# (c) derived per-genome weights
ax = axes[1, 0]
colors = ["C0"] * N_SIGNAL + ["C1"] * N_REDUNDANT
ax.bar(range(X.shape[1]), weights, color=colors, width=1.0)
ax.axhline(1.0, color="grey", ls="--", lw=1, label="w = 1 (unweighted)")
ax.set_xlabel("genome"); ax.set_ylabel("weight $w_j$")
ax.set_title("(c) Inverse-abundance weights\n"
             "(blue = 6 signal, orange = 60 redundant)")
ax.legend()

# (d) ARI before/after
ax = axes[1, 1]
x = np.arange(2)
ax.bar(x - 0.2, [ari_u_t, ari_u_s], 0.4, label="unweighted")
ax.bar(x + 0.2, [ari_w_t, ari_w_s], 0.4, label="weighted")
ax.set_xticks(x); ax.set_xticklabels(["true t", "spurious s"])
ax.set_ylabel("Adjusted Rand Index"); ax.set_ylim(-0.1, 1.05)
ax.axhline(0, color="k", lw=0.5)
ax.set_title("(d) Recovered partition"); ax.legend()

plt.tight_layout()
fig.savefig(HERE / "weighted_nem_results.png", dpi=110)
print(f"saved {HERE / 'weighted_nem_results.png'}")
