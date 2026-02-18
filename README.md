# NEM — Neighborhood EM for Spatial Clustering

NEM is a spatial clustering algorithm that combines the EM algorithm with
Hidden Markov Random Fields. Given a graph where each node carries a feature
vector, NEM produces a partition that accounts for both the data and the
spatial structure of the graph.

**Reference:**
Ambroise, C., Dang, V.M. and Govaert, G. (1997). Clustering of spatial data
by the EM algorithm. *geoENV I — Geostatistics for Environmental Applications*,
Vol. 9, Kluwer Academic Publisher, pp. 493–504.

## Repository structure

```
NEM/
├── csrc/           # Original C implementation (v1.07)
├── pynem/          # Python reimplementation (standalone package)
│   ├── src/pynem/  # Package source
│   └── tests/      # Test suite
├── examples/       # Example data (essai dataset)
└── README.md
```

---

## pynem — Python package

`pynem` is a pure-Python reimplementation of the NEM algorithm, with a
scikit-learn-style API. It uses NetworkX for graph handling, NumPy/SciPy for
computation, and Matplotlib for visualization.

### Installation

```bash
cd pynem
pip install -e ".[dev]"
```

### Quick start

```python
import pynem

# Load data (reads .str, .dat, .nei files)
G = pynem.io.read_graph("examples/essai")

# Fit NEM with 3 clusters and spatial regularization
model = pynem.NEM(n_clusters=3, beta=1.0, family="normal")
model.fit(G)

# Results
model.labels_        # hard classification (N,)
model.membership_    # soft classification (N, K)
model.centers_       # cluster centers (K, D)
model.dispersions_   # cluster dispersions (K, D)
model.proportions_   # cluster proportions (K,)
model.criteria_      # dict with U, D, G, L, M values

# Visualization
pynem.viz.plot_graph_clusters(G, model.labels_)
pynem.viz.plot_convergence(model.history_)
```

### Features

| Feature | Options |
|---|---|
| Algorithm | `nem` (mean field), `ncem` (ICM hard), `gem` (Gibbs) |
| Distributions | `normal`, `laplace`, `bernoulli` |
| Dispersion models | `s__` (shared), `sk_` (per-class), `s_d` (per-variable), `skd` (full) |
| Proportions | `p_` (equal), `pk` (free) |
| Beta estimation | `fix`, `psgrad` (pseudo-gradient), `heu_d`, `heu_l` (heuristics) |
| Initialization | `sort`, `random` (with multiple starts) |
| Missing data | `replace` (EM-style) or `ignore` |

### API

The `NEM` class follows scikit-learn conventions:

```python
model = pynem.NEM(
    n_clusters=3,           # number of clusters K
    beta=1.0,               # spatial regularization (0 = standard EM)
    algorithm="nem",        # nem | ncem | gem
    family="normal",        # normal | laplace | bernoulli
    dispersion="s__",       # s__ | sk_ | s_d | skd
    proportion="pk",        # p_ | pk
    beta_mode="fix",        # fix | psgrad | heu_d | heu_l
    init="sort",            # sort | random
    n_init=1,               # random restarts (for init="random")
    max_iter=100,
    tol=1e-3,
    random_state=42,
)
model.fit(G)
```

### Criteria

- **U** = D + 0.5 * beta * G — NEM criterion (main objective)
- **D** — Hathaway discrepancy (data fit)
- **G** — geographic cohesion (spatial smoothness)
- **L** — mixture log-likelihood
- **M** — Markovian pseudo-likelihood

---

## C implementation

The original C code (v1.07, 1999) by Van Mo Dang.

### Building

```bash
cd csrc && make exe
```

Produces `nem_exe` and utilities (`geo2nei`, `randord`, `err2`, `tcpu`).

### Usage

```bash
csrc/nem_exe examples/essai 3              # 3 clusters, default settings
csrc/nem_exe examples/essai 3 -b 0         # pure EM (no spatial effect)
csrc/nem_exe examples/essai 3 -b 0.1       # moderate spatial smoothing
csrc/nem_exe examples/essai 3 -B heu_d     # automatic beta estimation
csrc/nem_exe examples/essai 3 -a ncem      # hard classification (ICM)
csrc/nem_exe examples/essai 3 -m norm pk sk_  # free proportions, per-class variance
```

### Input files

NEM expects files sharing the same base name:

| File | Format |
|---|---|
| `.str` | `[Type] [N] [D]` — Type: `S` (spatial), `I` (image), `N` (non-spatial) |
| `.dat` | N x D matrix of feature vectors (space-separated) |
| `.nei` | Adjacency list: first line = weighted flag, then `[NodeID] [NbNeighbors] [Neighbors...]` |

### Output files

| File | Content |
|---|---|
| `.cf` | Hard classification (one label per node) |
| `.uf` | Fuzzy classification (K membership probabilities per node) |
| `.mf` | Model parameters (criteria, beta, centers, dispersions, proportions) |
| `.log` | Iteration log (with `-l y`) |

### Options

```bash
csrc/nem_exe -h options    # full option reference
```

| Option | Description |
|---|---|
| `-b <beta>` | Spatial strength (default 1.0, 0 = pure EM) |
| `-B <mode>` | Beta estimation: `fix`, `psgrad`, `heu_d`, `heu_l` |
| `-a <algo>` | Algorithm: `nem`, `ncem`, `gem` |
| `-m <fam> <prop> <disp>` | Model: family, proportions, dispersion |
| `-s <init> <param>` | Initialization: `s` (sort), `r` (random with N starts) |
| `-C <crit>` | Selection criterion: `U`, `M`, `D`, `L` |
| `-f fuzzy` | Output fuzzy membership |
| `-c <test> <thr>` | Convergence: `clas`, `crit`, `none` |
| `-i <max>` | Maximum iterations |
| `-l y` | Enable log file |

## License

MIT
