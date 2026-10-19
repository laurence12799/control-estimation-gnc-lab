"""Deterministic, component-scoped random-number streams for simulations."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.random import Generator


class NamedRandomStreams:
    """Provide stable random streams derived from a run seed and component name.

    Stable names, rather than component creation order, ensure that adding an
    unrelated sensor does not alter a pre-existing sensor's noise realization.
    """

    def __init__(self, root_seed: int) -> None:
        if root_seed < 0:
            raise ValueError("The root seed must be non-negative.")
        self._root_seed = root_seed
        self._generators: dict[str, Generator] = {}

    def generator(self, name: str) -> Generator:
        """Return the persistent generator assigned to a component name."""
        if not name:
            raise ValueError("A random stream name must not be empty.")
        if name not in self._generators:
            digest = hashlib.blake2b(
                f"{self._root_seed}:{name}".encode(),
                digest_size=16,
            ).digest()
            entropy = np.frombuffer(digest, dtype=np.uint32).tolist()
            self._generators[name] = np.random.default_rng(np.random.SeedSequence(entropy))
        return self._generators[name]
