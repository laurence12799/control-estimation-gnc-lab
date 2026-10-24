"""@file components.py
@brief Define small reusable infrastructure components.

@details These components support simulation assembly without implementing a
user-owned model, controller, estimator, or GNC algorithm.
"""

from __future__ import annotations

from typing import Any

from gnc_lab.sim.types import ModelSnapshot


class IdentityInputBuilder:
    """@brief Map actuator output directly to the plant input.

    Models that need to combine actuation with wind, loads, or other exogenous
    effects can provide their own input builder without changing the engine.
    """

    def build(self, time: float, actuation: Any, snapshot: ModelSnapshot) -> Any:
        """@brief Return actuator output without modification.

        @param time float: Current simulation boundary time [s]; unused.
        @param actuation Any: Actuator output to pass to the plant.
        @param snapshot ModelSnapshot: Current plant snapshot; unused.
        @return Any: ``actuation`` unchanged.
        """
        del time, snapshot
        return actuation
