"""Threading layer and thread limits for pynem's parallel kernels."""

try:
    import numba
except ImportError:
    numba = None


def _forksafe_layer():
    """Take a fork-safe numba threading layer unless one was asked for."""
    if numba is not None and numba.config.THREADING_LAYER == "default":
        numba.config.THREADING_LAYER = "forksafe"
