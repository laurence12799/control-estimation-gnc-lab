"""@file types.py
@brief Define numerical types shared by GNC Lab infrastructure.

@details This module deliberately contains no domain state definitions.
Individual models own their state and input meaning; the simulation kernel only
requires a finite one-dimensional floating-point state vector for numerical
integration.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatVector = NDArray[np.float64]


def as_float_vector(value: ArrayLike, *, name: str) -> FloatVector:
    """@brief Convert a value to an owned, finite, one-dimensional float vector.

    @param value ArrayLike: Numerical value to validate and convert.
    @param name str: Human-readable label used in validation-error messages.
    @return FloatVector: Independent ``float64`` copy of ``value``.
    @raises ValueError: If ``value`` is not one-dimensional or contains a
        non-finite element.
    """
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; received shape {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector.copy()
