"""Small infrastructure components that do not encode engineering algorithms."""

from __future__ import annotations

from typing import Any

from gnc_lab.sim.types import ModelSnapshot


class IdentityInputBuilder:
    """Use actuator output directly as the model input.

    Models that need to combine actuation with wind, loads, or other exogenous
    effects can provide their own input builder without changing the engine.
    """

    def build(self, time: float, actuation: Any, snapshot: ModelSnapshot) -> Any:
        """Return actuation unchanged."""
        del time, snapshot
        return actuation
