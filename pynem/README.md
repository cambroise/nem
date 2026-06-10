# pynem — Neighborhood EM for spatial clustering on graphs

[![Tests](https://github.com/cambroise/nem/actions/workflows/tests.yml/badge.svg)](https://github.com/cambroise/nem/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cambroise/nem/blob/main/pynem/LICENSE)

`pynem` is a standalone, pure-Python implementation of the **Neighborhood EM
(NEM)** algorithm: clustering that combines the EM algorithm with a **hidden
Markov random field**. Given a graph whose nodes carry feature vectors, NEM
produces a partition that accounts for both the data and the spatial structure
of the graph. It has a scikit-learn-style API and uses NumPy/SciPy, NetworkX and
(optionally) Numba.

It also faithfully reproduces the partitioning step of
[PPanGGOLiN](https://github.com/labgem/PPanGGOLiN) — the persistent / shell /
cloud partitioning of bacterial pangenomes is exactly this NEM algorithm — and
adds two original extensions: **per-variable weighting** (correcting genome
redundancy) and a **MAG-aware completeness** model (correcting genome
incompleteness).

## Installation

```bash
pip install pynem
```

Optional extras:
- `pynem[viz]` — **matplotlib** for the plotting utilities (`pynem.viz`).
  Without it, `import pynem` and all clustering still work; only plotting needs it.
- `pynem[fast]` — **Numba** (JIT acceleration of the sequential E-step; a
  pure-Python fallback is used otherwise).
- `pynem[dev]` — the test suite.

(They combine: `pip install "pynem[fast,viz]"`.)

## Quick start

```python
import numpy as np
import networkx as nx
from pynem import NEM

# A graph whose nodes carry feature vectors, plus a feature matrix X (N, D)
G = nx.path_graph(100)
X = np.random.default_rng(0).normal(size=(100, 2))

model = NEM(n_clusters=3, beta=1.0, family="normal")
model.fit(X, graph=G)

model.labels_        # hard classification (N,)
model.membership_    # soft classification (N, K)
model.centers_       # cluster centers (K, D)
model.criteria_      # dict with U, D, G, L, M
```

![NEM results on an SBM graph](https://raw.githubusercontent.com/cambroise/nem/main/examples/sbm_100_2_results.png)

## Pangenome partitioning (PPanGGOLiN-faithful)

```python
from pynem import partition_pangenome

# presence: (n_families, n_genomes) array of {0, 1}; graph: contiguity nx.Graph
res = partition_pangenome(presence, graph, K=3, beta=2.5)
res["partition"]   # array of "P" / "S" / "C" per gene family
```

On PPanGGOLiN's real *Chlamydia* test set (53 genomes, 1086 gene families),
`partition_pangenome` reproduces PPanGGOLiN **exactly** — identical
persistent / shell / cloud composition (871 / 56 / 159, agreement 1.000) and
soft membership matching the reference C core to ~5e-4 — while running a bit
faster (it stays in memory).

## Original extensions

- **Weighted NEM** — `feature_weights` (per variable) / `genome_weights` (per
  genome) down-weight redundant features so the partition reflects biology, not
  sampling. `pynem.genome_weights(...)` derives them automatically (Jaccard +
  UPGMA, silhouette).
- **MAG-aware completeness** — `completeness=` (per genome, from CheckM or
  `"auto"` self-estimation) forgives absences in incomplete metagenome-assembled
  genomes, restoring a persistent class that the standard model would collapse.

Both are **opt-in**: with the defaults, `pynem` reproduces standard NEM /
PPanGGOLiN byte-for-byte.

## Features

| Feature | Options |
|---|---|
| Algorithm | `nem` (mean field), `ncem` (ICM), `gem` (Gibbs) |
| Distributions | `normal`, `laplace`, `bernoulli` |
| Dispersion | `s__`, `sk_`, `s_d`, `skd` |
| Proportions | `p_` (equal), `pk` (free) |
| Beta | fixed, pseudo-gradient, heuristics |
| Init | `sort`, `random`, `param` |
| Site update | `parallel` (Jacobi), `seq` (Gauss-Seidel) |
| Feature weights | per-variable `feature_weights` (weighted NEM) |
| Completeness (MAG) | per-genome `completeness` (CheckM or `"auto"`) |

## Documentation & source

Full documentation, examples and the C reference implementation:
**https://github.com/cambroise/nem**

## References

- Ambroise, C., Dang, V. M., & Govaert, G. (1997). *Clustering of spatial data by
  the EM algorithm.* geoENV I — Geostatistics for Environmental Applications.
- Ambroise, C., & Govaert, G. (1998). *Convergence proof of an EM-type algorithm
  for spatial clustering.* Pattern Recognition Letters.

## License

MIT
