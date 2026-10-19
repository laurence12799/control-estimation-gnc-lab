"""Tests for recorded setup, signals, and optional component-state provenance."""

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
    StatefulEstimator,
    StatefulPlanner,
)


def test_result_retains_configuration_initial_conditions_and_signal_history() -> None:
    """Post-processing has the required run setup and boundary-by-boundary data."""
    planner = StatefulPlanner()
    controller = StatefulController()
    estimator = StatefulEstimator()
    sensor = NoiseSensor(noise_standard_deviation=0.0)
    actuator = BiasActuator(bias=0.25)
    config = SimulationConfig(start_time=0.0, stop_time=0.2, plant_dt=0.1)
    metadata = SimulationMetadata(
        name="provenance",
        seed=17,
        revision="abc123",
        tags=("test", "history"),
        attributes={"scenario": "recording"},
    )
    plan = SimulationPlan(
        model=ScalarInputTestDouble(),
        initial_state=np.array([2.0]),
        config=config,
        metadata=metadata,
        initial_command=-1.0,
        planner=TimedComponent(planner, PeriodicSchedule(0.1), initial_condition={"phase": 3}),
        controller=TimedComponent(
            controller, PeriodicSchedule(0.1), initial_condition={"mode": "auto"}
        ),
        estimator=estimator,
        estimator_initial_condition={"mean": 2.0},
        sensors=(
            SensorBinding(
                "position",
                sensor,
                PeriodicSchedule(0.1),
                initial_condition={"bias": 0.0},
            ),
        ),
        actuator=actuator,
        actuator_initial_condition={"bias": 0.25},
    )

    result = SimulationEngine().run(plan)

    assert result.config == config
    assert result.metadata == metadata
    np.testing.assert_array_equal(result.initial_conditions.plant_state, np.array([2.0]))
    assert result.initial_conditions.command == -1.0
    assert result.initial_conditions.planner == {"phase": 3}
    assert result.initial_conditions.controller == {"mode": "auto"}
    assert result.initial_conditions.estimator == {"mean": 2.0}
    assert result.initial_conditions.actuator == {"bias": 0.25}
    assert result.initial_conditions.sensors == {"position": {"bias": 0.0}}

    assert result.states.shape == (3, 1)
    np.testing.assert_allclose(result.states[:, 0], [2.0, 2.125, 2.35])
    first_record = result.records[0]
    second_record = result.records[1]
    assert first_record.output == {"value": 2.0}
    assert first_record.command == 1.0
    assert first_record.actuation == 1.25
    assert first_record.plant_input == 1.25
    assert first_record.reference == 1.5
    assert first_record.measurements["position"].value == 2.0
    assert first_record.component_states["planner"] == {
        "initial_condition": {"phase": 3},
        "updates": 1,
    }
    assert first_record.component_states["controller"] == {
        "initial_condition": {"mode": "auto"},
        "updates": 1,
    }
    assert first_record.component_states["estimator"] == {
        "initial_condition": {"mean": 2.0},
        "propagations": 0,
        "updates": 1,
    }
    assert first_record.component_states["sensors"]["position"] == {
        "initial_condition": {"bias": 0.0},
        "samples": 1,
    }
    assert first_record.component_states["actuator"] == {
        "applications": 1,
        "initial_condition": {"bias": 0.25},
    }
    assert second_record.component_states["estimator"]["propagations"] == 1
