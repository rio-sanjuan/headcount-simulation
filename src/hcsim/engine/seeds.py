"""The seed manager: one random generator per subsystem and trajectory.

Each generator is derived from the run seed, a stable hash of the subsystem's name, and the
trajectory index. A trajectory's streams therefore depend only on its index, not on execution
order or on how many trajectories run, so trajectory k is reproducible on its own. Adding a
subsystem never perturbs another's stream.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(name: str) -> int:
    """A process-independent 63-bit hash. Python's built-in hash is salted per process."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") >> 1


class SeedManager:
    def __init__(self, seed: int | None) -> None:
        self.seed = seed
        self._cache: dict[tuple[int, str], np.random.Generator] = {}

    def generator(self, trajectory: int, subsystem: str) -> np.random.Generator:
        """The cached generator for `subsystem` in `trajectory`. It advances step to step.

        Unseeded runs draw fresh entropy, so nothing about them is reproducible.
        """
        key = (trajectory, subsystem)
        if key not in self._cache:
            if self.seed is None:
                self._cache[key] = np.random.default_rng()
            else:
                sequence = np.random.SeedSequence([self.seed, stable_hash(subsystem), trajectory])
                self._cache[key] = np.random.default_rng(sequence)
        return self._cache[key]

    @property
    def reproducible(self) -> bool:
        return self.seed is not None
