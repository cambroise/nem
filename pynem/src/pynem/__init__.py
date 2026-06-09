"""pynem — Neighborhood EM for spatial clustering on graphs."""

from .core import NEM
from . import io, viz, models, spatial, metrics, ppanggolin, weights
from .ppanggolin import partition_pangenome
from .weights import genome_weights, jaccard_upgma_labels, abundance_weights

__version__ = "0.5.0"
__all__ = ["NEM", "io", "viz", "models", "spatial", "metrics",
           "ppanggolin", "partition_pangenome", "weights",
           "genome_weights", "jaccard_upgma_labels", "abundance_weights"]
