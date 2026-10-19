"""Tests for multi-rate scheduling on the fixed plant-time grid."""

from __future__ import annotations

import numpy as np

from gnc_lab.sim.engine import SimulationEngine
from gnc_lab.sim.types import (
    PeriodicSchedule,
    SensorBinding,
    SimulationConfig,
    SimulationMetadata,
    SimulationPlan,
    TimedComponent,
)
from tests.simulation_doubles import (
    BiasActuator,
    NoiseSensor,
    ScalarInputTestDouble,
    StatefulController,
    StatefulPlanner,
)


def test_multiple_component_rates_follow_their_own_aligned_periods() -> None:
    """Sensors, planner, and controller fire only at their requested boundaries."""
    sensor = NoiseSensor()
    planner = StatefulPlanner()
    controller = StatefulController()
    actuator = BiasActuator()
    result = SimulationEngine().run(
        SimulationPlan(
            model=ScalarInputTestDouble(),
            initial_state=np.array([0.0]),
            config=SimulationConfig(start_time=0.0, stop_time=0.6, plant_dt=0.05),
            metadata=SimulationMetadata(name="multi-rate schedule"),
            sensors=(SensorBinding("fast", sensor, PeriodicSchedule(period=0.1)),),
            planner=TimedComponent(planner, PeriodicSchedule(period=0.2, phase=0.1)),
            controller=TimedComponent(controller, PeriodicSchedule(period=0.3)),
            actuator=actuator,
        )
    )

    np.testing.assert_allclose(
        [record.time for record in result.records if "fast" in record.measurements],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    np.testing.assert_allclose(planner.update_times, [0.1, 0.3, 0.5])
    np.testing.assert_allclose(controller.update_times, [0.0, 0.3, 0.6])
    np.testing.assert_allclose(actuator.application_times, [0.0, 0.3, 0.6])


def test_multirate_scheduling_works_with_another_plant_time_step() -> None:
    """Rates are expressed in time, rather than baked into one chosen plant dt."""
    planner = StatefulPlanner()
    controller = StatefulController()
    SimulationEngine().run(
        SimulationPlan(
            model=ScalarInputTestDouble(),
            initial_state=np.array([0.0]),
            config=SimulationConfig(start_time=0.0, stop_time=0.4, plant_dt=0.02),
            metadata=SimulationMetadata(name="alternate multi-rate schedule"),
            planner=TimedComponent(planner, PeriodicSchedule(period=0.04)),
            controller=TimedComponent(controller, PeriodicSchedule(period=0.1)),
        )
    )

    np.testing.assert_allclose(
        planner.update_times,
        [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4],
    )
    np.testing.assert_allclose(controller.update_times, [0.0, 0.1, 0.2, 0.3, 0.4])
