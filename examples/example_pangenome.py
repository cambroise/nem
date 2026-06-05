"""Example: pangenome partitioning à la PPanGGOLiN with ``partition_pangenome``.

Simulates a pangenome (gene families x genomes presence/absence + a genomic
contiguity graph), partitions the families into persistent / shell / cloud with
pynem's PPanGGOLiN-faithful pipeline, and saves a figure for the README.

This uses only pynem (no PPanGGOLiN install required).
"""

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pynem" / "src"))

from pynem import partition_pangenome

HERE = Path(__file__).parent

# ── Simulate a pangenome ─────────────────────────────────────────────────────

rng = np.random.default_rng(0)
N_PERS, N_SHELL, N_CLOUD = 60, 90, 120
N_ORG = 40
labels_true = np.array([0] * N_PERS + [1] * N_SHELL + [2] * N_CLOUD)
N = len(labels_true)
PCS = np.array(["P", "S", "C"])
truth = PCS[labels_true]


def presence(label):
    if label == 0:      # persistent: present in almost every genome
        p = rng.uniform(0.90, 1.0)
    elif label == 1:    # shell: intermediate frequency
        p = rng.uniform(0.25, 0.75)
    else:               # cloud: rare
        p = rng.uniform(0.01, 0.15)
    return (rng.random(N_ORG) < p).astype(int)


X = np.vstack([presence(l) for l in labels_true]).astype(float)

# genomic contiguity graph: same-category families tend to be neighbours
G = nx.Graph()
G.add_nodes_from(range(N))
for c in range(3):
    idx = np.where(labels_true == c)[0]
    for a, b in zip(idx, np.roll(idx, 1)):
        G.add_edge(int(a), int(b), weight=1.0)
    for _ in range(len(idx)):
        a, b = rng.choice(idx, 2, replace=False)
        if a != b:
            G.add_edge(int(a), int(b), weight=1.0)
for _ in range(30):                       # a little cross-category noise
    a, b = map(int, rng.integers(0, N, 2))
    if a != b:
        G.add_edge(a, b, weight=1.0)

# ── Partition (persistent / shell / cloud) ───────────────────────────────────

res = partition_pangenome(X, G, K=3, beta=2.5, sm_degree=10, max_iter=100)
partition = res["partition"]

from collections import Counter
print("Partition:", dict(Counter(partition)))
agree = float(np.mean(partition == truth))
print(f"Agreement with simulated truth: {agree:.3f}")

# ── Visualization ────────────────────────────────────────────────────────────

color = {"P": "C0", "S": "C1", "C": "C2"}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# presence/absence matrix, rows ordered by partition
ax = axes[0]
order = np.argsort([{"P": 0, "S": 1, "C": 2}[p] for p in partition])
ax.imshow(X[order], aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_xlabel("Genome")
ax.set_ylabel("Gene family (ordered by partition)")
ax.set_title("Presence / absence")

# family graph coloured by partition
ax = axes[1]
pos = nx.spring_layout(G, seed=1, k=0.15)
nx.draw_networkx_edges(G, pos, alpha=0.08, ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=35,
                       node_color=[color[p] for p in partition], ax=ax)
ax.set_axis_off()
ax.set_title("Partitioned pangenome graph")
handles = [plt.Line2D([0], [0], marker="o", ls="", color=color[c],
                      label={"P": "persistent", "S": "shell", "C": "cloud"}[c])
           for c in ["P", "S", "C"]]
ax.legend(handles=handles, loc="upper right")

fig.suptitle(f"pynem partition_pangenome — agreement = {agree:.2f}",
             fontsize=14, y=1.02)
fig.savefig(str(HERE / "pangenome_results.png"), dpi=150, bbox_inches="tight")
print("Saved: pangenome_results.png")
plt.show()
