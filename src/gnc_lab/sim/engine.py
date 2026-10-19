"""Fixed-step orchestration for composable closed-loop simulations."""

from __future__ import annotations

import copy
from typing import Any

from gnc_lab.core.types import FloatVector, as_float_vector
from gnc_lab.sim.components import IdentityInputBuilder
from gnc_lab.sim.integrators import Integrator, RK4Integrator
from gnc_lab.sim.randomness import NamedRandomStreams
from gnc_lab.sim.types import (
    Measurement,
    ModelSnapshot,
    SimulationInitialConditions,
    SimulationPlan,
    SimulationResult,
    StepRecord,
)


class SimulationEngine:
    """Run a fixed-step simulation with optional scheduled components.

    At each plant boundary, the engine samples due sensors, updates the
    estimator, planner, and controller, applies actuation, records the result,
    then advances the plant state through one zero-order-held RK4 interval.
    """

    def __init__(self, integrator: Integrator | None = None) -> None:
        self._integrator = integrator or RK4Integrator()

    def run(self, plan: SimulationPlan) -> SimulationResult:
        """Execute a validated simulation plan and return its in-memory history."""
        self._validate_plan(plan)
        streams = NamedRandomStreams(plan.metadata.seed)
        self._reset_components(plan, streams)

        state = as_float_vector(plan.initial_state, name="initial_state")
        default_input = _copy_payload(plan.model.default_input())
        command = (
            _copy_payload(plan.initial_command)
            if plan.initial_command is not None
            else default_input
        )
        actuation: Any = None
        plant_input = default_input
        initial_conditions = self._initial_conditions(plan, state, command)

        reference: Any = None
        feedback: Any = None
        records: list[StepRecord] = []

        for step_index in range(plan.config.number_of_steps() + 1):
            time = plan.config.time_at(step_index)
            snapshot = self._snapshot(plan, time, state, plant_input)
            measurements = self._sample_sensors(plan, step_index, snapshot, streams)

            if plan.estimator is not None and measurements:
                plan.estimator.update(time, measurements)
            feedback = plan.estimator.estimate() if plan.estimator is not None else snapshot

            if plan.planner is not None and plan.planner.schedule.is_due(
                step_index,
                plan.config.plant_dt,
            ):
                reference = plan.planner.component.update(time, feedback)

            command_changed = False
            if plan.controller is not None and plan.controller.schedule.is_due(
                step_index,
                plan.config.plant_dt,
            ):
                command = plan.controller.component.update(time, reference, feedback)
                command_changed = True
            elif plan.controller is None and plan.command_source is not None:
                command = plan.command_source.command(time, feedback)
                command_changed = True

            if step_index == 0 or command_changed:
                actuation = self._apply_actuator(plan, time, command, streams)
            plant_input = self._build_input(plan, time, actuation, snapshot)
            recorded_snapshot = self._snapshot(plan, time, state, plant_input)
            records.append(
                StepRecord(
                    time=time,
                    state=recorded_snapshot.state.copy(),
                    output=_copy_payload(recorded_snapshot.output),
                    command=_copy_payload(command),
                    actuation=_copy_payload(actuation),
                    plant_input=_copy_payload(plant_input),
                    feedback=_copy_payload(feedback),
                    reference=_copy_payload(reference),
                    measurements={
                        name: Measurement(
                            sensor_name=measurement.sensor_name,
                            time=measurement.time,
                            value=_copy_payload(measurement.value),
                        )
                        for name, measurement in measurements.items()
                    },
                    component_states=self._component_states(plan),
                )
            )

            if step_index == plan.config.number_of_steps():
                break

            state = self._integrator.step(
                plan.model.derivative,
                time,
                state,
                plan.config.plant_dt,
                plant_input,
            )
            if plan.estimator is not None:
                plan.estimator.propagate(time, plan.config.plant_dt, plant_input)

        return SimulationResult(
            config=plan.config,
            metadata=plan.metadata,
            initial_conditions=initial_conditions,
            records=tuple(records),
        )

    @staticmethod
    def _validate_plan(plan: SimulationPlan) -> None:
        if plan.controller is not None and plan.command_source is not None:
            raise ValueError("Configure either a controller or a command_source, not both.")

        names = [binding.name for binding in plan.sensors]
        if len(set(names)) != len(names):
            raise ValueError("Each sensor binding must have a unique name.")

        schedules = [binding.schedule for binding in plan.sensors]
        if plan.planner is not None:
            schedules.append(plan.planner.schedule)
        if plan.controller is not None:
            schedules.append(plan.controller.schedule)
        for schedule in schedules:
            schedule.is_due(0, plan.config.plant_dt)

    @staticmethod
    def _reset_components(plan: SimulationPlan, streams: NamedRandomStreams) -> None:
        if plan.planner is not None:
            plan.planner.component.reset(
                plan.planner.initial_condition,
                streams.generator("planner"),
            )
        if plan.controller is not None:
            plan.controller.component.reset(
                plan.controller.initial_condition,
                streams.generator("controller"),
            )
        if plan.estimator is not None:
            plan.estimator.reset(
                plan.estimator_initial_condition,
                streams.generator("estimator"),
            )
        if plan.actuator is not None:
            plan.actuator.reset(
                plan.actuator_initial_condition,
                streams.generator("actuator"),
            )
        for binding in plan.sensors:
            binding.sensor.reset(
                binding.initial_condition,
                streams.generator(f"sensor:{binding.name}"),
            )

    @staticmethod
    def _snapshot(
        plan: SimulationPlan,
        time: float,
        state: FloatVector,
        plant_input: Any,
    ) -> ModelSnapshot:
        truth_state = as_float_vector(state, name="model state")
        output = plan.model.output(time, truth_state.copy(), plant_input)
        return ModelSnapshot(
            time=time,
            state=truth_state,
            plant_input=_copy_payload(plant_input),
            output=_copy_payload(output),
        )

    @staticmethod
    def _sample_sensors(
        plan: SimulationPlan,
        step_index: int,
        snapshot: ModelSnapshot,
        streams: NamedRandomStreams,
    ) -> dict[str, Measurement]:
        measurements: dict[str, Measurement] = {}
        for binding in plan.sensors:
            if binding.schedule.is_due(step_index, plan.config.plant_dt):
                measurements[binding.name] = Measurement(
                    sensor_name=binding.name,
                    time=snapshot.time,
                    value=binding.sensor.sample(
                        snapshot,
                        streams.generator(f"sensor:{binding.name}"),
                    ),
                )
        return measurements

    @staticmethod
    def _apply_actuator(
        plan: SimulationPlan,
        time: float,
        command: Any,
        streams: NamedRandomStreams,
    ) -> Any:
        if plan.actuator is None:
            return _copy_payload(command)
        return plan.actuator.apply(time, command, streams.generator("actuator"))

    @staticmethod
    def _build_input(
        plan: SimulationPlan,
        time: float,
        actuation: Any,
        snapshot: ModelSnapshot,
    ) -> Any:
        builder = plan.input_builder or IdentityInputBuilder()
        return builder.build(time, actuation, snapshot)

    @staticmethod
    def _initial_conditions(
        plan: SimulationPlan,
        state: FloatVector,
        command: Any,
    ) -> SimulationInitialConditions:
        return SimulationInitialConditions(
            plant_state=state.copy(),
            command=_copy_payload(command),
            planner=_copy_payload(plan.planner.initial_condition)
            if plan.planner is not None
            else None,
            controller=(
                _copy_payload(plan.controller.initial_condition)
                if plan.controller is not None
                else None
            ),
            estimator=_copy_payload(plan.estimator_initial_condition),
            actuator=_copy_payload(plan.actuator_initial_condition),
            sensors={
                binding.name: _copy_payload(binding.initial_condition) for binding in plan.sensors
            },
        )

    @staticmethod
    def _component_states(plan: SimulationPlan) -> dict[str, Any]:
        states: dict[str, Any] = {}
        if plan.planner is not None:
            states["planner"] = _optional_snapshot(plan.planner.component)
        if plan.controller is not None:
            states["controller"] = _optional_snapshot(plan.controller.component)
        if plan.estimator is not None:
            states["estimator"] = _optional_snapshot(plan.estimator)
        if plan.actuator is not None:
            states["actuator"] = _optional_snapshot(plan.actuator)
        if plan.sensors:
            states["sensors"] = {
                binding.name: _optional_snapshot(binding.sensor) for binding in plan.sensors
            }
        return states


def _copy_payload(value: Any) -> Any:
    """Copy logged or boundary data so later component mutation cannot rewrite history."""
    return copy.deepcopy(value)


def _optional_snapshot(component: Any) -> Any:
    """Read optional component state without imposing a snapshot implementation."""
    snapshot = getattr(component, "snapshot", None)
    return _copy_payload(snapshot()) if callable(snapshot) else None
