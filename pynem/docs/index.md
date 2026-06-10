# pynem

**Neighborhood EM (NEM)** — clustering that combines the EM algorithm with a
**hidden Markov random field**. Given a graph whose nodes carry feature vectors,
NEM produces a partition that accounts for both the data and the spatial
structure of the graph. Scikit-learn-style API; NumPy / SciPy / NetworkX, and
optionally Numba.

It also faithfully reproduces the partitioning step of
[PPanGGOLiN](https://github.com/labgem/PPanGGOLiN) (persistent / shell / cloud
pangenome partitioning) and adds two original extensions: **per-variable
weighting** (genome redundancy) and a **MAG-aware completeness** model (genome
incompleteness).

## Installation

```bash
pip install pynem            # core
pip install "pynem[fast]"    # + Numba JIT (sequential E-step)
pip install "pynem[viz]"     # + matplotlib (plotting)
```

## Quick start

```python
import numpy as np, networkx as nx
from pynem import NEM

G = nx.path_graph(100)
X = np.random.default_rng(0).normal(size=(100, 2))

model = NEM(n_clusters=3, beta=1.0, family="normal")
model.fit(X, graph=G)

model.labels_        # hard classification (N,)
model.membership_    # soft classification (N, K)
model.predict(Xnew)  # classify new feature vectors
```

## Pangenome partitioning

```python
from pynem import partition_pangenome

res = partition_pangenome(presence, graph, K=3, beta=2.5)
res["partition"]     # "P" / "S" / "C" per gene family

# correct genome redundancy (weighting) and/or incompleteness (completeness):
w = pynem.genome_weights(presence, selection="silhouette")
res = partition_pangenome(presence, graph, genome_weights=w, completeness="auto")
```

See the [API reference](api.md) for the full documentation, and the
[GitHub repository](https://github.com/cambroise/nem) for examples, the maths
notes, and the original C implementation.
