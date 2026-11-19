"""Thread limits for pynem's parallel kernels and the BLAS under NumPy."""

try:
    import numba
except ImportError:
    numba = None

from threadpoolctl import threadpool_limits


def _forksafe_layer():
    """Take a fork-safe numba threading layer unless one was asked for."""
    if numba is not None and numba.config.THREADING_LAYER == "default":
        numba.config.THREADING_LAYER = "forksafe"


def set_threads(n):
    """Cap pynem's kernels and the BLAS under NumPy at ``n`` threads.

    :param n: Maximum number of threads, clamped to at least 1 and to numba's build maximum.
    :return: The numba thread count in force, or None without numba.
    """
    n = max(1, int(n))
    _forksafe_layer()
    threadpool_limits(n)
    if numba is None:
        return None
    n = min(n, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(n)
    return n
