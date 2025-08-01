"""
ucurv.backend
=============

Backend, so use ncp as a stand in for np basically, will either call numpy or cupy behind the scenes, as their methods are mostly interchangeable.
Must explicitly specify cupy if want to use, and will only use cupy if GPU is available, or else will throw an error.

* `_backend` can be "auto" (default), "numpy", or "cupy".
* `set_backend(name)` must be called **before** the first use of :func:`ncp`.
* `ncp()` returns the active array module (`numpy` or `cupy`).

Typical use
-----------
    from ucurv.backend import ncp, set_backend

    y = ncp().sin(ncp().pi / 4)
"""

import numpy as np
import warnings

def get_module(backend: str = "numpy"):
    """Returns correct numerical module based on backend string

    Parameters
    ----------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if backend == "auto" or backend == "numpy":
        ncp = np
    elif backend == "cupy":
        try:
            ncp = __import__("cupy")
        except ImportError:
            warnings.warn(
                "engine='cupy' requested, but CuPy is not installed, falling back to numpy",
                category=ImportWarning,
                stacklevel=2
            )
            ncp = np
    else:
        raise ValueError("backend must be numpy, or cupy")
    return ncp