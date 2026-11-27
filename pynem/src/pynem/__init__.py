"""pynem — Neighborhood EM for spatial clustering on graphs."""

from . import io, metrics, models, ppanggolin, spatial, threads, viz, weights
from .core import NEM
from .ppanggolin import partition_pangenome
from .threads import set_threads
from .weights import abundance_weights, genome_weights, jaccard_upgma_labels

__version__ = "0.6.1"
__all__ = ["NEM", "io", "viz", "models", "spatial", "metrics",
           "ppanggolin", "partition_pangenome", "weights",
           "genome_weights", "jaccard_upgma_labels", "abundance_weights",
           "threads", "set_threads"]
