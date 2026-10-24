"""@file randomness.py
@brief Provide deterministic component-scoped random streams for simulations.

@details Stable stream names make stochastic component behavior independent of
component construction and request order.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.random import Generator


class NamedRandomStreams:
    """@brief Manage stable random streams derived from a run seed and name.

    Stable names, rather than component creation order, ensure that adding an
    unrelated sensor does not alter a pre-existing sensor's noise realization.
    """

    def __init__(self, root_seed: int) -> None:
        """@brief Initialize a named random-stream collection.

        @param root_seed int: Non-negative run-level random seed.
        @raises ValueError: If ``root_seed`` is negative.
        @note The instance retains created generators so repeated requests for
        the same name continue the same stream.
        """
        if root_seed < 0:
            raise ValueError("The root seed must be non-negative.")
        self._root_seed = root_seed
        self._generators: dict[str, Generator] = {}

    def generator(self, name: str) -> Generator:
        """@brief Return the persistent generator assigned to a stream name.

        @param name str: Stable, non-empty component or experiment identifier.
        @return Generator: Generator deterministically derived from the root
            seed and ``name``.
        @raises ValueError: If ``name`` is empty.
        """
        if not name:
            raise ValueError("A random stream name must not be empty.")
        if name not in self._generators:
            digest = hashlib.blake2b(
                f"{self._root_seed}:{name}".encode(),
                digest_size=16,
            ).digest()
            entropy = np.frombuffer(digest, dtype=np.uint32).tolist()
            self._generators[name] = np.random.default_rng(
                np.random.SeedSequence(entropy)
            )
        return self._generators[name]
