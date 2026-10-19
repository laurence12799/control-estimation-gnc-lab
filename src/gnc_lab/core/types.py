"""Numerical types shared by infrastructure modules.

This module deliberately contains no domain state definitions. Individual
models own their state and input meaning; the simulation kernel only requires a
finite one-dimensional floating-point state vector for numerical integration.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatVector = NDArray[np.float64]


def as_float_vector(value: ArrayLike, *, name: str) -> FloatVector:
    """Return a copied finite vector, rejecting ambiguous numerical shapes."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector.copy()
