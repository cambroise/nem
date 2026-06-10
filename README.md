# NEM — Neighborhood EM for Spatial Clustering

NEM is a spatial clustering algorithm that combines the EM algorithm with
Hidden Markov Random Fields. Given a graph where each node carries a feature
vector, NEM produces a partition that accounts for both the data and the
spatial structure of the graph.

>The original C code  was written by  [Mô Van Dang](https://www.linkedin.com/in/m%C3%B4-dang-884b4950/) in 1996 and the last C version dates from 1999 (v1.07). Claude Code assisted in resurrecting the C code and developing a Python implementation. 


The model, algorithm, and original code served as a foundation for PanGGOLiN (Partitioned PanGenome Graph Of Linked Neighbors), a bioinformatics software package dedicated to pangenome analysis—the complete set of genes found within a group of related organisms. Initiated in 2016, the project supports large-scale comparative analysis of bacterial genomes, with a focus on genome evolution, gene diversity, and the identification of stable and variable genomic regions. PPanGGOLiN relies on graph-based representations to model gene co-localization relationships across genomes ([Paggolin paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007732), [PPanGGOLiN GitHub Repository](https://github.com/labgem/PPanGGOLiN))



## References

- Ambroise, C., Dang, V. M., & Govaert, G. (1997).  
  **Clustering of spatial data by the EM algorithm.**  
  *geoENV I — Geostatistics for Environmental Applications*, 9, 493–504.  
  [PDF](https://ambroise.cloud/publications/chap_ambroise1997clustering/chap_ambroise1997clustering.pdf)

- Ambroise, C., & Govaert, G. (1998).
  **Convergence Proof of an EM-type Algorithm for Spatial Clustering.**  
*Pattern Recognition Letters*.  
  [PDF](https://ambroise.cloud/publications/arti_ambroise1998convergence/arti_ambroise1998convergence.pdf)



## Repository structure

```
NEM/
├── csrc/           # Original C implementation (v1.07)
├── pynem/          # Python reimplementation (standalone package, v0.6.0)
│   ├── src/pynem/  # Package source
│   └── tests/      # Test suite
├── ppanggolin/     # PPanGGOLiN submodule (embeds the same NEM core)
├── examples/       # Example data and synthetic generators
│   ├── generate.py           # SBMData, PottsImageData generators
│   ├── example_sbm_100.py    # SBM graph example (100 nodes, 2D)
│   ├── example_sbm_200.py    # SBM graph example (200 nodes, 2D)
│   ├── example_potts_30x30.py # Potts image example (30×30 grid, 2D)
│   ├── example_pangenome.py  # Pangenome partitioning (persistent/shell/cloud)
│   ├── example_weighted.py   # Weighted NEM (genome redundancy correction)
│   └── essai.*               # Original dataset (100 nodes, 3 vars)
└── README.md
```

---

## Example: SBM graph with 100 nodes

This example generates a Stochastic Block Model graph with 100 nodes,
3 classes, and 2-dimensional Gaussian emissions, then clusters it with NEM.

### Generate synthetic data

```python
from generate import SBMData

sbm = SBMData(
    n=100, k=3, d=2,
    p_in=0.2, p_out=0.02,
    centers=[[0, 0], [3, 3], [0, 3]],
    sigma=1.0, seed=42,
)
sbm.export("sbm_100_2")   # writes .str, .dat, .nei, .true.cf
```

### Run NEM and evaluate

```python
import pynem

G = pynem.io.read_graph("examples/sbm_100_2")
model = pynem.NEM(n_clusters=3, beta=1.0, family="normal")
model.fit(G)

# Adjusted Rand Index (requires true labels)
true_labels = sbm.labels + 1  # convert to 1-based
ari = pynem.metrics.adjusted_rand_index(true_labels, model.labels_)
print(f"Adjusted Rand Index: {ari:.4f}")
# Adjusted Rand Index: 0.9403
```

### Visualize results

```python
fig = pynem.viz.plot_results(G, model, true_labels=true_labels,
                             var_names=["Variable 1", "Variable 2"])
```

The top row shows estimated vs true labels; the bottom row shows each
dimension of the emission vector on the graph:

![NEM results on SBM 100 nodes](examples/sbm_100_2_results.png)

Convergence of the NEM criterion U:

![Convergence](examples/sbm_100_2_convergence.png)

On this dataset NEM recovers the true partition with high accuracy
(ARI = 0.94) in 7 iterations.

---

## Example: Potts image with 30×30 grid

This example generates a Potts Markov Random Field on a 30×30 pixel grid with
3 classes and 2-dimensional Gaussian emissions, then clusters it with NEM.

### Generate synthetic data

```python
from generate import PottsImageData

potts = PottsImageData(
    nl=30, nc=30, k=3, beta=0.8,
    centers=[[0, 0], [3, 3], [0, 3]],
    sigma=1.0, seed=42,
)
potts.export("potts_30x30_3")   # writes .str, .dat, .nei, .true.cf
```

### Run NEM and evaluate

```python
import pynem

G = pynem.io.read_graph("examples/potts_30x30_3")
model = pynem.NEM(n_clusters=3, beta=1.0, family="normal")
model.fit(G)

# Adjusted Rand Index (requires true labels)
true_labels = potts.labels + 1  # convert to 1-based
ari = pynem.metrics.adjusted_rand_index(true_labels, model.labels_)
print(f"Adjusted Rand Index: {ari:.4f}")
# Adjusted Rand Index: 0.7379
```

### Visualize results

```python
fig = pynem.viz.plot_results(G, model, true_labels=true_labels,
                             var_names=["Variable 1", "Variable 2"])
```

The top row shows estimated vs true labels; the bottom row shows each
dimension of the emission vector as a pixel image:

![NEM results on Potts 30×30 grid](examples/potts_30x30_3_results.png)

Convergence of the NEM criterion U:

![Convergence](examples/potts_30x30_3_convergence.png)

On this dataset NEM recovers the main spatial structure (ARI = 0.74),
the lower score compared to the SBM example reflecting the harder
segmentation task: the classes have overlapping emissions (σ = 1,
center gap = 3) and the Potts prior creates irregular, non-convex regions.

---

## pynem — Python package

`pynem` is a pure-Python reimplementation of the NEM algorithm, with a
scikit-learn-style API. It uses NetworkX for graph handling, NumPy/SciPy for
computation, and Matplotlib for visualization.

### Installation

```bash
cd pynem
pip install -e .
```

Optional extras: `.[fast]` adds **Numba** (JIT acceleration of the sequential
E-step on large graphs; a pure-Python fallback is used otherwise), `.[dev]`
adds the test suite (pytest) — e.g. `pip install -e ".[fast]"`.

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
fig = pynem.viz.plot_results(G, model)
pynem.viz.plot_convergence(model.history_)
```

### Pangenome partitioning (PPanGGOLiN)

[PPanGGOLiN](https://github.com/labgem/PPanGGOLiN) partitions prokaryotic
pangenomes into **persistent / shell / cloud** gene families — and its
partitioning step *is* the NEM algorithm. `pynem.partition_pangenome`
reproduces that pipeline (Bernoulli model, `K=3`, `beta` rescaling, `sm_degree`
hub filter, sequential E-step, P/S/C labelling), validated element-wise against
the reference NEM C core embedded in PPanGGOLiN.

```python
from pynem import partition_pangenome

# presence: (n_families, n_genomes) array of {0,1}; graph: contiguity nx.Graph
res = partition_pangenome(presence, graph, K=3, beta=2.5)
res["partition"]   # array of "P"/"S"/"C" per gene family
res["membership"]  # soft classification (n_families, K)
res["centers"]     # binary modes mu (K, n_genomes)
res["criteria"]    # U, D, G, L, M
```

Run `python examples/example_pangenome.py` to reproduce the figure below
(persistent / shell / cloud on a simulated pangenome, agreement 0.99 with the
ground truth):

![Pangenome partitioning](examples/pangenome_results.png)

#### Real data: same results as PPanGGOLiN

On PPanGGOLiN's official test dataset (53 *Chlamydia* genomes → **1086 gene
families**), `pynem` reproduces PPanGGOLiN **exactly**. Running both on the same
NEM input, the partitions are identical — same persistent / shell / cloud
composition and the same label for *every* gene family (agreement **1.000**) —
and pynem's soft membership matches the reference C core to **~5e-4** (the
residual is float32 in the C core vs float64 in pynem):

| | Persistent | Shell | Cloud |
|---|---:|---:|---:|
| PPanGGOLiN (production workflow) | 871 | 56 | 159 |
| pynem `partition_pangenome` | 871 | 56 | 159 |

![Chlamydia pangenome partition](examples/chlam_pangenome_presence.png)

This is not an artefact of simulated data: pynem *is* PPanGGOLiN's partitioning
step, down to the soft membership, on a real bacterial pangenome.

#### Performance vs the C core

`pynem` runs the same NEM as PPanGGOLiN's embedded C core and is competitive on
speed. On simulated pangenomes (50 genomes), with the Numba-accelerated
sequential E-step, it is consistently a bit faster than `run_partitioning` —
the C core's wall-clock time includes the disk I/O the pipeline performs to
write and re-read its NEM files, whereas pynem stays in memory. Best-of-3
timings:

| Gene families N | PPanGGOLiN (C core + I/O) | pynem (Python + Numba) | speed-up |
|---:|---:|---:|---:|
| 1 000 | 0.039 s | 0.020 s | 1.9× |
| 2 000 | 0.069 s | 0.044 s | 1.6× |
| 4 000 | 0.180 s | 0.077 s | 2.3× |
| 8 000 | 0.250 s | 0.155 s | 1.6× |
| 16 000 | 0.494 s | 0.310 s | 1.6× |

![Speed: pynem vs PPanGGOLiN](examples/speed_ppanggolin_vs_pynem.png)

Both scale linearly with N (it is the same algorithm). Without Numba, pynem
falls back to pure Python and is slower — the JIT is what makes the sequential
sweep competitive with native C.

### Weighted NEM: down-weighting redundant genomes

Genomes are rarely a balanced sample: an over-sampled clade (say 50 strains of
one species) weighs as many times as it has representatives and **biases the
partition toward its own biology**. pynem can correct this by giving each
genome (each feature column) a weight `w_j`: a redundant genome gets a small
weight, a rare one a large weight. Only the E-step changes — the closed-form
M-step is preserved (see the maths note). Weighting is **opt-in**: with no
weights you get the standard NEM, *byte-for-byte identical* (verified), so the
PPanGGOLiN reproduction above is untouched.

```python
import pynem
from pynem import NEM, partition_pangenome, genome_weights

# 1. derive weights automatically: group the genomes (columns) by Jaccard
#    distance + UPGMA, then turn group sizes into inverse-abundance weights.
#    Number of groups: fixed (n_groups, default 10) or chosen by silhouette.
w = pynem.genome_weights(presence, selection="silhouette")   # or n_groups=10

# 2. pass them to the pangenome pipeline (or to NEM via feature_weights=w)
res = partition_pangenome(presence, graph, K=3, beta=2.5, genome_weights=w)
```

**Hierarchical classification of the genomes.** `genome_weights` clusters the
genomes with Jaccard distance and UPGMA (`scipy`), picking the number of groups
by the silhouette criterion (no scikit-learn needed). Run
`python examples/example_weighted.py` for a self-contained demo where 6 *signal*
genomes carry the true gene partition and 60 *redundant* genomes carry an
unrelated one: the silhouette finds exactly two genome groups, down-weights the
redundant block (w ≈ 0.55 vs 5.5), and the weighted NEM **recovers the true
partition** (ARI 0.96) where the unweighted one locks onto the redundant signal
(ARI 0.00):

![Weighted NEM on a redundancy scenario](examples/weighted_nem_results.png)

On the real *Chlamydia* genomes, the same Jaccard + UPGMA step exposes the
sub-structure of the 53 strains — large near-identical groups are down-weighted
(w ≈ 0.4), isolated genomes up-weighted (w ≈ 6):

![Hierarchical classification of Chlamydia genomes](examples/chlam_genome_dendrogram.png)

> Caveat: deriving the weights from the presence/absence matrix is convenient
> but **circular** (the weights come from the very matrix being partitioned). A
> distance independent of gene content — e.g. ANI — is preferable when
> available; the weighting mechanism is identical, only the source of `w_j`
> changes.

### MAG-aware partitioning: completeness (v0.5.0)

Metagenome-assembled genomes (MAGs) are often **incomplete**: a truly persistent
gene is frequently *absent* simply because the assembly missed it, which
artificially shrinks the persistent class. Following
[mOTUpan](https://doi.org/10.1093/nargab/lqac060) (Buck et al. 2022), pynem can
take a per-genome **completeness** `gamma_j in (0, 1]` (e.g. from CheckM) and
*forgive* absences in incomplete genomes — the Bernoulli presence probability
becomes `mu_kj * gamma_j`, so an absence costs little where the genome is known
incomplete. This is **distinct from** and **combinable with** the redundancy
weighting above (completeness fixes incompleteness, weighting fixes over-sampling).

```python
from pynem import partition_pangenome

# completeness from CheckM (one value per genome)
res = partition_pangenome(presence, graph, K=3, beta=2.5, completeness=gamma)

# or self-estimate it from the data (no CheckM) — length-seed init + iterative
# re-estimation from the inferred persistent set (mOTUpan Eq. 6)
res = partition_pangenome(presence, graph, K=3, beta=2.5, completeness="auto")
res["completeness"]          # the completeness vector used / estimated
```

On the real *Chlamydia* pangenome eroded to 40–70 % completeness, the standard
pipeline **collapses** the persistent class (871 → 3 families); with
`completeness` (given *or* self-estimated) it is **restored** (all 871 recovered;
the self-estimated `gamma` correlates 0.98 with the true completeness).
`completeness=None` (default) keeps the standard, PPanGGOLiN-faithful behaviour
unchanged.

### Evaluation with simulated data

When true labels are available (e.g. from `SBMData` or `PottsImageData`),
compute the Adjusted Rand Index to quantify clustering accuracy:

```python
ari = pynem.metrics.adjusted_rand_index(true_labels, model.labels_)
```

### Features

| Feature | Options |
|---|---|
| Algorithm | `nem` (mean field), `ncem` (ICM hard), `gem` (Gibbs) |
| Distributions | `normal`, `laplace`, `bernoulli` |
| Dispersion models | `s__` (shared), `sk_` (per-class), `s_d` (per-variable), `skd` (full) |
| Proportions | `p_` (equal), `pk` (free) |
| Beta estimation | `fix`, `psgrad` (pseudo-gradient), `heu_d`, `heu_l` (heuristics) |
| Initialization | `sort`, `random` (multiple starts), `param` (from given parameters) |
| Site update | `parallel` (Jacobi), `seq` (sequential Gauss-Seidel, NEM default) |
| Missing data | `replace` (EM-style) or `ignore` |
| Feature weights | `feature_weights` per variable (weighted NEM); `None` = standard NEM |
| Completeness (MAG) | `completeness` per genome (Bernoulli; CheckM array or `"auto"`); `None` = standard |

Robustness: empty classes are reinitialised k-means++ style (centre moved to the
point farthest from the dominant class), and the sequential E-step is
JIT-compiled with Numba when available (pure-Python fallback otherwise).

`fit` validates its inputs up front and raises a clear `ValueError`/`TypeError`
instead of failing deep in the pipeline: a graph is required (`fit(X, graph=...)`
or a single feature-carrying `nx.Graph`), its nodes must be the contiguous
integers `0..N-1`, the node count must match `X`, and `1 <= K <= N`. Internally,
the previously overloaded `EPSILON` is split into named floors with explicit
intent — `PROB_FLOOR` (probabilities/proportions), `VAR_FLOOR` (dispersions),
`ZERO_DISP_TOL` (degenerate point-mass dimensions) and `DIV_GUARD` (division
guards); the values are unchanged, so the PPanGGOLiN reproduction and the
example results above are preserved exactly. The per-iteration `log(p_k f_k)` is
also computed once and shared between the E-step and the criteria.

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
    init="sort",            # sort | random | param
    site_update="parallel", # parallel (Jacobi) | seq (Gauss-Seidel)
    n_init=1,               # random restarts (for init="random")
    max_iter=100,
    tol=1e-3,
    random_state=42,
)
model.fit(G)
```

### Visualization

- `pynem.viz.plot_results(G, model, true_labels=...)` — estimated labels, true labels, and per-dimension feature views
- `pynem.viz.plot_labels(G, labels)` — labels on graph or image
- `pynem.viz.plot_feature(G, values)` — continuous values on graph or image
- `pynem.viz.plot_convergence(history)` — criterion vs iteration
- `pynem.viz.plot_membership(G, membership)` — soft membership heatmap
- `pynem.viz.plot_cluster_centers(centers)` — parallel coordinates

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
| `.str` | `S N D` or `N N D` (spatial/non-spatial) or `I Nl Nc D` (image grid) |
| `.dat` | N x D matrix of feature vectors (space-separated) |
| `.nei` | Adjacency list: first line = weighted flag, then `[NodeID] [NbNeighbors] [Neighbors...]` |


where 
- N is the sample size
- D is the number of features of each sample vector
- Nl is the number of lines of the image (if the imput data is an image)
- Nc is the number of columns of the image (if eh input data is an image)

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
