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

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Literal

# ---------------------------------------------------------------------------
# INTERNAL STATE
# ---------------------------------------------------------------------------
_BackendLiteral = Literal["auto", "numpy", "cupy"]

_backend: _BackendLiteral = os.getenv("UCURV_BACKEND", "auto").lower()  # user hint
_backend_module: ModuleType | None = None                               # cache


def _resolve_backend() -> ModuleType:
    """Resolve once, cache in `_backend_module`, then return it."""
    global _backend_module

    if _backend_module is not None:            # already chosen
        return _backend_module

    if _backend == "numpy":
        mod = importlib.import_module("numpy")

    elif _backend == "cupy":
        mod = importlib.import_module("cupy")  # will raise if missing

    else:  # "auto"  → prefer Numpy 
        try:
            mod = importlib.import_module("numpy")
        except ModuleNotFoundError:
            raise Exception("Numpy is not installed")

    _backend_module = mod
    return mod


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def ncp() -> ModuleType:     # noqa:  N802  (keep NumPy‑style name)
    """Return NumPy or CuPy according to the active back end."""
    return _resolve_backend()


def get_backend() -> _BackendLiteral:
    """Return the current backend flag ("auto", "numpy", or "cupy")."""
    return _backend


def set_backend(name: _BackendLiteral) -> None:
    """
    Set the back end flag **before** the first call to :func:`ncp`.

    Parameters
    ----------
    name : {"auto", "numpy", "cupy"}
    """
    global _backend
    if name not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'cupy'")

    if _backend_module is not None:
        raise RuntimeError(
            "Backend already resolved; set UCURV_BACKEND or call "
            "set_backend() before importing ucurv modules that use ncp()."
        )

    _backend = name

set_backend("cupy")