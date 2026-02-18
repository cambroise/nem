# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The NEM (Neighborhood EM) algorithm performs spatial clustering by combining the EM algorithm with Hidden Markov Random Fields. It clusters nodes of a graph where each node carries a feature vector (D-dimensional), using both the feature data and the graph topology (neighborhood structure) to produce spatially coherent clusters.

### Goals

1. **Make the C code functional** — compile and run `nem_exe` and its utilities.
2. **Develop a Python package (`pynem`)** — a standalone library reimplementing NEM using standard Python modules for graph handling and visualization.

## Algorithm Summary

At each iteration:
- **M-step**: estimate model parameters (class centers, dispersions, proportions, beta) from current soft/hard classification.
- **E-step**: update classification using both data likelihood and spatial neighborhood weights:
  ```
  c_ik ∝ p_k * f_k(x_i) * exp(beta * sum_j∈N(i) c_jk)
  ```
  where `N(i)` are the neighbors of node `i` in the graph.

### Supported variants
| Feature | Options |
|---|---|
| Algorithm | NEM (mean field), NCEM (ICM hard), GEM (Gibbs) |
| Distributions | Normal, Laplace, Bernoulli |
| Dispersion models | S__ (shared), SK_ (per-class), S_D (per-variable), SKD (full) |
| Proportions | P_ (equal), Pk (free) |
| Beta estimation | Fixed, pseudo-gradient, heuristic-D, heuristic-L |
| Initialization | Sort, random, mixture-EM, file, partial labels |
| Missing data | Replace (EM-style) or ignore |

### Criteria
- **U** (NEM criterion) = D + 0.5*beta*G — main objective
- **D** (Hathaway/discrepancy) — data fit
- **G** (geographic cohesion) — spatial smoothness
- **L** (mixture log-likelihood)
- **M** (Markovian pseudo-likelihood)

## Data Formats

### Structure file (`.str`)
```
S 100 3
```
`[Type] [N] [D]` — Type: `S`=spatial graph, `I`=image grid; N=number of nodes; D=number of variables per node.

### Data file (`.dat`)
```
5.534  4.509  4.033
3.405  5.395  6.334
...
```
N rows x D columns, space/tab-separated floats. Each row is the feature vector of a graph node.

### Neighborhood file (`.nei`)
```
0 32 3 4 9 10 ...
1 41 6 10 16 18 ...
```
Each line: `[NodeID] [NumNeighbors] [Neighbor1] [Neighbor2] ...` — defines the graph adjacency list.

## Architecture

### C code — `src/`

| File | Role |
|---|---|
| `nem_exe.c` | Main program, I/O, entry point (`mainfunc`) |
| `nem_arg.c` | Command-line argument parsing (20+ options) |
| `nem_alg.c` | Core NEM algorithm (E-step, M-step, convergence, beta estimation) |
| `nem_mod.c` | Statistical model estimation (Normal, Laplace, Bernoulli densities) |
| `nem_nei.c` | Neighborhood system (image grid or irregular graph) |
| `nem_typ.h` | All type definitions (`DataT`, `StatModelT`, `SpatialT`, `NemParaT`, `CriterT`) |
| `nem_rnd.c` | Random number generation |
| `nem_hlp.c` | Help/usage text |
| `lib_io.c/h` | General I/O utilities (file parsing, token reading) |
| `genmemo.h` | Memory allocation macros |
| `geo2nei.c` | Utility: geographic coordinates to `.nei` file |
| `makefile` | Build: `make exe` compiles all targets |

#### Building

```bash
cd src && make exe
```
Produces: `nem_exe`, `geo2nei`, `randord`, `err2`, `tcpu`.

#### Running

```bash
./nem_exe essai 3          # cluster essai.dat into 3 classes using essai.nei graph
./nem_exe essai 3 -b 2.0   # with beta=2
./nem_exe essai 3 -a ncem  # hard classification (ICM)
```

### Python package — `pynem/`

The Python reimplementation should be a **standalone installable package** with this structure:

```
pynem/
├── pyproject.toml
├── src/
│   └── pynem/
│       ├── __init__.py
│       ├── core.py          # NEM algorithm (E-step, M-step, convergence)
│       ├── models.py        # Distribution families (Normal, Laplace, Bernoulli)
│       ├── spatial.py       # Graph/neighborhood handling
│       ├── io.py            # Read/write .dat, .nei, .str files
│       └── viz.py           # Visualization utilities
└── tests/
    └── test_nem.py
```

#### Key design decisions

- **Graph representation**: use `networkx.Graph` — each node has a feature vector attribute (numpy array of length D). NetworkX is the standard Python library for graph manipulation.
- **Numerical computation**: `numpy` and `scipy` for array operations, density functions, and statistics.
- **Visualization**: `matplotlib` for plotting clusters on graphs (node coloring by class), convergence curves, and class parameter distributions. For spatial/image data, support 2D scatter plots and grid heatmaps.
- **No Cython/C bindings** — pure Python reimplementation for clarity and maintainability.
- **API style**: scikit-learn compatible where possible (`fit()`, `predict()`, `labels_`, `n_clusters`).

#### Core API sketch

```python
import networkx as nx
from pynem import NEM

# Load data
G = pynem.io.read_graph("essai.nei", "essai.dat", "essai.str")
# G is a networkx.Graph where G.nodes[i]['features'] is a numpy array of shape (D,)

# Fit model
model = NEM(n_clusters=3, beta=1.0, family="normal", algorithm="nem")
model.fit(G)

# Results
model.labels_          # hard classification (N,)
model.membership_      # soft classification (N, K)
model.centers_         # class centers (K, D)
model.dispersions_     # class dispersions (K, D)
model.proportions_     # class proportions (K,)
model.criteria_        # dict with U, D, G, L, M values

# Visualization
pynem.viz.plot_graph_clusters(G, model.labels_)
pynem.viz.plot_convergence(model.history_)
```

#### Dependencies

- `numpy`
- `scipy`
- `networkx`
- `matplotlib`

## Example data

- `essai.str` — 100 nodes, 3 variables, spatial graph
- `essai.dat` — 100x3 feature matrix
- `essai.nei` — irregular graph with ~30-40 neighbors per node

## References

- Ambroise, C., Dang, V.M., and Govaert, G. (1997). *Clustering of spatial data by the EM algorithm*. geoENV I — Geostatistics for Environmental Applications.
- C implementation v1.07 (1999) by Van Mo Dang, UTC, URA CNRS 817.
