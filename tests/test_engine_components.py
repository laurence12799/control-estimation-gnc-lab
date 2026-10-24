"""Matrix tests for every valid optional-component simulation assembly."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any

import numpy as np
import pytest

from gnc_lab.sim.engine import SimulationEngine
from gnc_lab.sim.types import (
    ModelSnapshot,
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
    StatefulCommandSource,
    StatefulController,
    StatefulEstimator,
    StatefulPlanner,
)


def _component_permutations() -> list[tuple[bool, bool, bool, bool, bool, bool]]:
    """@brief Return all valid optional-component inclusion combinations.

    @return list[tuple[bool, bool, bool, bool, bool, bool]]: Boolean flags
        for sensor, estimator, planner, controller, actuator, and source.
    @note Controller and command source are intentionally mutually exclusive.
    """
    return [
        permutation
        for permutation in product((False, True), repeat=6)
        if not (permutation[3] and permutation[5])
    ]


@pytest.mark.parametrize(
    (
        "include_sensor",
        "include_estimator",
        "include_planner",
        "include_controller",
        "include_actuator",
        "include_source",
    ),
    _component_permutations(),
)
def test_every_valid_optional_component_assembly_runs(
    include_sensor: bool,
    include_estimator: bool,
    include_planner: bool,
    include_controller: bool,
    include_actuator: bool,
    include_source: bool,
) -> None:
    """The engine composes all 48 valid on/off component combinations."""
    model = ScalarInputTestDouble()
    planner = StatefulPlanner()
    controller = StatefulController()
    estimator = StatefulEstimator()
    actuator = BiasActuator()
    command_source = StatefulCommandSource()
    sensor = NoiseSensor()

    plan = SimulationPlan(
        model=model,
        initial_state=np.array([0.0]),
        config=SimulationConfig(start_time=0.0, stop_time=0.2, plant_dt=0.1),
        metadata=SimulationMetadata(name="component combination", seed=9),
        planner=(
            TimedComponent(planner, PeriodicSchedule(period=0.2))
            if include_planner
            else None
        ),
        controller=(
            TimedComponent(controller, PeriodicSchedule(period=0.2))
            if include_controller
            else None
        ),
        estimator=estimator if include_estimator else None,
        sensors=(SensorBinding("test-sensor", sensor, PeriodicSchedule(period=0.1)),)
        if include_sensor
        else (),
        actuator=actuator if include_actuator else None,
        command_source=command_source if include_source else None,
    )

    result = SimulationEngine().run(plan)

    assert len(result.records) == 3
    _assert_component_history(
        result.records[-1].component_states,
        include_sensor=include_sensor,
        include_estimator=include_estimator,
        include_planner=include_planner,
        include_controller=include_controller,
        include_actuator=include_actuator,
    )
    if include_sensor:
        assert all("test-sensor" in record.measurements for record in result.records)
    else:
        assert all(not record.measurements for record in result.records)
    if include_controller:
        np.testing.assert_allclose(controller.update_times, [0.0, 0.2])
    elif include_source:
        np.testing.assert_allclose(command_source.call_times, [0.0, 0.1, 0.2])


def test_truth_snapshot_is_used_for_feedback_when_no_estimator_is_configured() -> None:
    """Truth feedback remains explicit rather than masquerading as an estimate."""
    controller = StatefulController()
    planner = StatefulPlanner()
    SimulationEngine().run(
        SimulationPlan(
            model=ScalarInputTestDouble(),
            initial_state=np.array([0.0]),
            config=SimulationConfig(start_time=0.0, stop_time=0.1, plant_dt=0.1),
            metadata=SimulationMetadata(name="truth feedback"),
            planner=TimedComponent(planner, PeriodicSchedule(period=0.1)),
            controller=TimedComponent(controller, PeriodicSchedule(period=0.1)),
        )
    )

    assert all(
        isinstance(feedback, ModelSnapshot) for feedback in planner.feedback_history
    )
    assert all(
        isinstance(feedback, ModelSnapshot) for feedback in controller.feedback_history
    )


def test_estimator_output_replaces_truth_feedback_when_an_estimator_is_configured() -> (
    None
):
    """A controller receives the estimator's published representation when present."""
    controller = StatefulController()
    estimator = StatefulEstimator()
    SimulationEngine().run(
        SimulationPlan(
            model=ScalarInputTestDouble(),
            initial_state=np.array([0.0]),
            config=SimulationConfig(start_time=0.0, stop_time=0.1, plant_dt=0.1),
            metadata=SimulationMetadata(name="estimate feedback"),
            estimator=estimator,
            controller=TimedComponent(controller, PeriodicSchedule(period=0.1)),
        )
    )

    assert all(isinstance(feedback, dict) for feedback in controller.feedback_history)
    assert estimator.propagation_count == 1


def test_command_source_and_controller_are_rejected_together() -> None:
    """Closed-loop and open-loop command ownership must remain unambiguous."""
    with pytest.raises(ValueError, match="either a controller or a command_source"):
        SimulationEngine().run(
            SimulationPlan(
                model=ScalarInputTestDouble(),
                initial_state=np.array([0.0]),
                config=SimulationConfig(start_time=0.0, stop_time=0.1, plant_dt=0.1),
                metadata=SimulationMetadata(name="invalid command ownership"),
                controller=TimedComponent(
                    StatefulController(), PeriodicSchedule(period=0.1)
                ),
                command_source=StatefulCommandSource(),
            )
        )


def _assert_component_history(
    component_states: Mapping[str, Any],
    *,
    include_sensor: bool,
    include_estimator: bool,
    include_planner: bool,
    include_controller: bool,
    include_actuator: bool,
) -> None:
    """@brief Assert provenance includes exactly the configured components.

    @param component_states Mapping[str, Any]: Recorded component snapshots.
    @param include_sensor bool: Whether sensor state is expected.
    @param include_estimator bool: Whether estimator state is expected.
    @param include_planner bool: Whether planner state is expected.
    @param include_controller bool: Whether controller state is expected.
    @param include_actuator bool: Whether actuator state is expected.
    """
    expected_names = {
        name
        for name, included in {
            "planner": include_planner,
            "controller": include_controller,
            "estimator": include_estimator,
            "actuator": include_actuator,
            "sensors": include_sensor,
        }.items()
        if included
    }
    assert set(component_states) == expected_names
