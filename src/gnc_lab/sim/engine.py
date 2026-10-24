"""@file engine.py
@brief Orchestrate fixed-step, composable closed-loop simulations.

@details The engine owns scheduling, signal routing, zero-order input holding,
component lifecycle calls, and result provenance. It does not own engineering
models or algorithms.
"""

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
    """@brief Run a fixed-step simulation with optional scheduled components.

    At each plant boundary, the engine samples due sensors, updates the
    estimator, planner, and controller, applies actuation, records the result,
    then advances the plant state through one zero-order-held RK4 interval.
    """

    def __init__(self, integrator: Integrator | None = None) -> None:
        """@brief Initialize the engine with an integration method.

        @param integrator Integrator | None: Optional one-step integrator. If
            omitted, the engine uses :class:`RK4Integrator`.
        """
        self._integrator = integrator or RK4Integrator()

    def run(self, plan: SimulationPlan) -> SimulationResult:
        """@brief Execute a plan and return its complete in-memory history.

        @param plan SimulationPlan: Validatable model, component, timing, and
            metadata assembly.
        @return SimulationResult: Ordered records and copied initial setup.
        @raises ValueError: If the plan has conflicting command sources,
            duplicate sensor names, or grid-misaligned component schedules.
        """
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
            feedback = (
                plan.estimator.estimate() if plan.estimator is not None else snapshot
            )

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
        """@brief Validate structural constraints before a simulation starts.

        @param plan SimulationPlan: Assembly to validate.
        @raises ValueError: If controller and command source coexist, sensor
            names are duplicated, or a schedule is not plant-grid aligned.
        """
        if plan.controller is not None and plan.command_source is not None:
            raise ValueError(
                "Configure either a controller or a command_source, not both."
            )

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
        """@brief Reset configured stateful components with named RNG streams.

        @param plan SimulationPlan: Component assembly to initialize.
        @param streams NamedRandomStreams: Run-scoped stream provider.
        @note Each component receives a stable, independent stream name.
        """
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
        """@brief Evaluate and copy plant truth at one boundary.

        @param plan SimulationPlan: Plan containing the model to evaluate.
        @param time float: Boundary time [s].
        @param state FloatVector: Truth state at ``time``.
        @param plant_input Any: Current input used to evaluate model output.
        @return ModelSnapshot: Copied state, input, and model output.
        """
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
        """@brief Sample every sensor due at the current plant boundary.

        @param plan SimulationPlan: Plan containing sensor bindings.
        @param step_index int: Zero-based plant-boundary index.
        @param snapshot ModelSnapshot: Truth data presented to each sensor.
        @param streams NamedRandomStreams: Run-scoped RNG provider.
        @return dict[str, Measurement]: Measurements keyed by sensor name.
        """
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
        """@brief Apply the optional actuator to a command.

        @param plan SimulationPlan: Plan containing the optional actuator.
        @param time float: Command-update time [s].
        @param command Any: Command to apply.
        @param streams NamedRandomStreams: Run-scoped RNG provider.
        @return Any: Actuation, or a deep copy of ``command`` without an
            actuator.
        """
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
        """@brief Build the model input from actuation and plant truth.

        @param plan SimulationPlan: Plan containing an optional input builder.
        @param time float: Boundary time [s].
        @param actuation Any: Current actuator output.
        @param snapshot ModelSnapshot: Current plant-truth data.
        @return Any: Input held for the next plant interval.
        """
        builder = plan.input_builder or IdentityInputBuilder()
        return builder.build(time, actuation, snapshot)

    @staticmethod
    def _initial_conditions(
        plan: SimulationPlan,
        state: FloatVector,
        command: Any,
    ) -> SimulationInitialConditions:
        """@brief Copy the initial setup to preserve run provenance.

        @param plan SimulationPlan: Assembly containing initialization data.
        @param state FloatVector: Validated initial truth state.
        @param command Any: Initial command selected for the run.
        @return SimulationInitialConditions: Deep-copied initial setup.
        """
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
                binding.name: _copy_payload(binding.initial_condition)
                for binding in plan.sensors
            },
        )

    @staticmethod
    def _component_states(plan: SimulationPlan) -> dict[str, Any]:
        """@brief Collect optional component snapshots at one boundary.

        @param plan SimulationPlan: Assembly containing recordable components.
        @return dict[str, Any]: Snapshots keyed by component role and sensor
            name.
        """
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
                binding.name: _optional_snapshot(binding.sensor)
                for binding in plan.sensors
            }
        return states


def _copy_payload(value: Any) -> Any:
    """@brief Deep-copy data crossing a simulation or logging boundary.

    @param value Any: Payload to copy.
    @return Any: Independent copy of ``value``.
    @note This private helper prevents later component mutation from rewriting
        result history.
    """
    return copy.deepcopy(value)


def _optional_snapshot(component: Any) -> Any:
    """@brief Read an optional component snapshot without requiring one.

    @param component Any: Component that may implement a callable
        ``snapshot`` method.
    @return Any: Deep-copied snapshot payload, or ``None`` when unavailable.
    """
    snapshot = getattr(component, "snapshot", None)
    return _copy_payload(snapshot()) if callable(snapshot) else None
