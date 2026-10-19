"""Tests for zero-order-held plant input in the fixed-step engine."""

from __future__ import annotations

import numpy as np

from gnc_lab.sim.engine import SimulationEngine
from gnc_lab.sim.types import (
    PeriodicSchedule,
    SimulationConfig,
    SimulationMetadata,
    SimulationPlan,
    TimedComponent,
)
from tests.simulation_doubles import ScalarInputTestDouble, StatefulController


def test_plant_input_is_held_for_every_rk4_stage_until_the_next_command_update() -> None:
    """A controller update cannot alter an input inside an RK4 plant interval."""
    model = ScalarInputTestDouble()
    controller = StatefulController()
    result = SimulationEngine().run(
        SimulationPlan(
            model=model,
            initial_state=np.array([0.0]),
            config=SimulationConfig(start_time=0.0, stop_time=0.4, plant_dt=0.1),
            metadata=SimulationMetadata(name="zero-order hold"),
            controller=TimedComponent(controller, PeriodicSchedule(period=0.2)),
        )
    )

    assert [record.command for record in result.records] == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert [record.plant_input for record in result.records] == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert model.rhs_inputs == [1.0] * 8 + [2.0] * 8
