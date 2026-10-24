"""@file types.py
@brief Define typed data structures for simulation assembly and results.

@details The classes in this module communicate timing, component bindings,
initial conditions, and immutable run history between the simulation engine and
user-owned engineering components.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from gnc_lab.core.types import FloatVector

if TYPE_CHECKING:
    from gnc_lab.sim.contracts import (
        Actuator,
        CommandSource,
        ContinuousModel,
        Controller,
        Estimator,
        InputBuilder,
        Planner,
        Sensor,
    )


@dataclass(frozen=True)
class SimulationMetadata:
    """@brief Store human-readable and reproducibility metadata for a run.

    @param name str: Non-empty descriptive run name.
    @param seed int: Non-negative root seed for deterministic random streams.
    @param revision str | None: Optional source-control revision identifier.
    @param tags tuple[str, ...]: Optional immutable labels for grouping runs.
    @param attributes Mapping[str, str]: Optional string key-value provenance.
    """

    name: str
    seed: int = 0
    revision: str | None = None
    tags: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """@brief Validate metadata fields after dataclass construction.

        @raises ValueError: If ``name`` is blank or ``seed`` is negative.
        """
        if not self.name.strip():
            raise ValueError("Simulation metadata requires a non-empty name.")
        if self.seed < 0:
            raise ValueError("Simulation metadata seed must be non-negative.")


@dataclass(frozen=True)
class SimulationConfig:
    """@brief Specify fixed-step timing for a numerical experiment.

    @param start_time float: First recorded boundary time [s].
    @param stop_time float: Final recorded boundary time [s].
    @param plant_dt float: Positive fixed plant-integration step [s].
    """

    start_time: float
    stop_time: float
    plant_dt: float

    def __post_init__(self) -> None:
        """@brief Validate timing values and their grid alignment.

        @raises ValueError: If the time range is invalid or the duration is not
            an integer number of plant steps.
        """
        if self.plant_dt <= 0.0:
            raise ValueError("plant_dt must be positive.")
        if self.stop_time <= self.start_time:
            raise ValueError("stop_time must be greater than start_time.")
        self.number_of_steps()

    def number_of_steps(self) -> int:
        """@brief Return the exact number of fixed plant steps.

        @return int: Number of integration intervals from start to stop.
        @raises ValueError: If the duration is not an integer multiple of
            ``plant_dt`` within floating-point tolerance.
        """
        raw_steps = (self.stop_time - self.start_time) / self.plant_dt
        steps = round(raw_steps)
        tolerance = max(1e-12, abs(raw_steps) * 1e-12)
        if not np.isclose(raw_steps, steps, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Simulation duration must be an integer multiple of plant_dt for the "
                "fixed-step engine."
            )
        return int(steps)

    def time_at(self, step_index: int) -> float:
        """@brief Return the time associated with a plant-grid boundary.

        @param step_index int: Zero-based integration-boundary index.
        @return float: Boundary time [s].
        """
        return self.start_time + step_index * self.plant_dt


@dataclass(frozen=True)
class PeriodicSchedule:
    """@brief Describe a periodic update schedule aligned to the plant grid.

    @param period float: Positive event period [s].
    @param phase float: First-event offset from simulation start [s], in the
        half-open interval ``[0, period)``.
    """

    period: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        """@brief Validate schedule period and phase.

        @raises ValueError: If the period is non-positive or the phase lies
            outside its valid interval.
        """
        if self.period <= 0.0:
            raise ValueError("Schedule period must be positive.")
        if self.phase < 0.0 or self.phase >= self.period:
            raise ValueError(
                "Schedule phase must be greater than or equal to zero and below period."
            )

    def is_due(self, step_index: int, plant_dt: float) -> bool:
        """@brief Determine whether an event is due at a plant-grid boundary.

        @param step_index int: Zero-based boundary index to evaluate.
        @param plant_dt float: Fixed plant integration step [s].
        @return bool: ``True`` when the event is due at ``step_index``.
        @raises ValueError: If period or phase is not grid-aligned.
        """
        period_steps = self._aligned_steps(self.period, plant_dt, "period")
        phase_steps = self._aligned_steps(self.phase, plant_dt, "phase")
        return (
            step_index >= phase_steps and (step_index - phase_steps) % period_steps == 0
        )

    @staticmethod
    def _aligned_steps(value: float, plant_dt: float, label: str) -> int:
        """@brief Convert a grid-aligned duration to an integer step count.

        @param value float: Duration or phase to convert [s].
        @param plant_dt float: Plant integration step [s].
        @param label str: Field label used in a validation-error message.
        @return int: Integer number of plant steps.
        @raises ValueError: If ``value`` is not a multiple of ``plant_dt``.
        """
        raw_steps = value / plant_dt
        steps = round(raw_steps)
        tolerance = max(1e-12, abs(raw_steps) * 1e-12)
        if not np.isclose(raw_steps, steps, rtol=0.0, atol=tolerance):
            raise ValueError(
                f"Schedule {label} must be an integer multiple of plant_dt."
            )
        return int(steps)


@dataclass(frozen=True)
class TimedComponent[ComponentT]:
    """@brief Bind a scheduled component to its initial condition.

    @param component ComponentT: Planner or controller implementation.
    @param schedule PeriodicSchedule: Update timing for ``component``.
    @param initial_condition Any: User-defined component initialization data.
    """

    component: ComponentT
    schedule: PeriodicSchedule
    initial_condition: Any = None


@dataclass(frozen=True)
class SensorBinding:
    """@brief Bind a uniquely named sensor to a sampling schedule.

    @param name str: Non-empty unique sensor identifier.
    @param sensor Sensor: Sensor implementation to invoke when due.
    @param schedule PeriodicSchedule: Sampling schedule for ``sensor``.
    @param initial_condition Any: User-defined sensor initialization data.
    """

    name: str
    sensor: Sensor
    schedule: PeriodicSchedule
    initial_condition: Any = None

    def __post_init__(self) -> None:
        """@brief Validate the public sensor identifier.

        @raises ValueError: If ``name`` is blank.
        """
        if not self.name.strip():
            raise ValueError("Sensor bindings require a non-empty name.")


@dataclass(frozen=True)
class ModelSnapshot:
    """@brief Capture truth information visible at one simulation boundary.

    @param time float: Snapshot time [s].
    @param state FloatVector: Truth plant state at ``time``.
    @param plant_input Any: Input used to evaluate the model output.
    @param output Any: Model-defined truth output at ``time``.
    """

    time: float
    state: FloatVector
    plant_input: Any
    output: Any


@dataclass(frozen=True)
class Measurement:
    """@brief Record one timestamped sensor observation.

    @param sensor_name str: Identifier of the producing sensor.
    @param time float: Sample time [s].
    @param value Any: Model-specific measurement payload.
    """

    sensor_name: str
    time: float
    value: Any


@dataclass(frozen=True)
class StepRecord:
    """@brief Record truth, signals, and component state at one boundary.

    @param time float: Recorded boundary time [s].
    @param state FloatVector: Truth plant state.
    @param output Any: Model-defined truth output.
    @param command Any: Command supplied by the controller or command source.
    @param actuation Any: Actuator output after actuator effects.
    @param plant_input Any: Zero-order-held input supplied to the plant.
    @param feedback Any: Truth snapshot or estimator output used for feedback.
    @param reference Any: Planner output held at this boundary.
    @param measurements Mapping[str, Measurement]: Due sensor measurements.
    @param component_states Mapping[str, Any]: Optional component snapshots.
    """

    time: float
    state: FloatVector
    output: Any
    command: Any
    actuation: Any
    plant_input: Any
    feedback: Any
    reference: Any
    measurements: Mapping[str, Measurement]
    component_states: Mapping[str, Any]


@dataclass(frozen=True)
class SimulationInitialConditions:
    """@brief Preserve copied initial conditions with a completed result.

    @param plant_state FloatVector: Initial truth state.
    @param command Any: Initial command before the first update.
    @param planner Any: Planner initial condition, if configured.
    @param controller Any: Controller initial condition, if configured.
    @param estimator Any: Estimator initial condition, if configured.
    @param actuator Any: Actuator initial condition, if configured.
    @param sensors Mapping[str, Any]: Initial conditions keyed by sensor name.
    """

    plant_state: FloatVector
    command: Any
    planner: Any
    controller: Any
    estimator: Any
    actuator: Any
    sensors: Mapping[str, Any]


@dataclass(frozen=True)
class SimulationResult:
    """@brief Hold the complete in-memory result of one simulation run.

    @param config SimulationConfig: Fixed-step timing used for the run.
    @param metadata SimulationMetadata: Run identity and reproducibility data.
    @param initial_conditions SimulationInitialConditions: Copied run setup.
    @param records tuple[StepRecord, ...]: Ordered boundary records.
    @note Large-scale on-disk result storage is intentionally outside this
        infrastructure layer.
    """

    config: SimulationConfig
    metadata: SimulationMetadata
    initial_conditions: SimulationInitialConditions
    records: tuple[StepRecord, ...]

    @property
    def times(self) -> np.ndarray:
        """@brief Return recorded integration-boundary times.

        @return np.ndarray: One-dimensional timestamp array [s].
        """
        return np.asarray([record.time for record in self.records], dtype=np.float64)

    @property
    def states(self) -> np.ndarray:
        """@brief Return the recorded truth-state history.

        @return np.ndarray: State matrix with one row per timestamp.
        """
        return np.stack([record.state for record in self.records])


@dataclass(frozen=True)
class SimulationPlan:
    """@brief Assemble user-owned components around the reusable kernel.

    @param model ContinuousModel: Plant model defining dynamics and outputs.
    @param initial_state FloatVector: Initial truth state for ``model``.
    @param config SimulationConfig: Fixed-step timing configuration.
    @param metadata SimulationMetadata: Run identity and random seed.
    @param initial_command Any: Command used before a command update.
    @param planner TimedComponent[Planner] | None: Optional reference planner.
    @param controller TimedComponent[Controller] | None: Optional controller.
    @param estimator Estimator | None: Optional state or output estimator.
    @param estimator_initial_condition Any: Estimator initialization payload.
    @param sensors tuple[SensorBinding, ...]: Optional sensor bindings.
    @param actuator Actuator | None: Optional discrete actuator model.
    @param actuator_initial_condition Any: Actuator initialization payload.
    @param command_source CommandSource | None: Optional open-loop command
        producer; mutually exclusive with ``controller``.
    @param input_builder InputBuilder | None: Optional builder that combines
        actuation with other input effects.
    """

    model: ContinuousModel
    initial_state: FloatVector
    config: SimulationConfig
    metadata: SimulationMetadata
    initial_command: Any = None
    planner: TimedComponent[Planner] | None = None
    controller: TimedComponent[Controller] | None = None
    estimator: Estimator | None = None
    estimator_initial_condition: Any = None
    sensors: tuple[SensorBinding, ...] = ()
    actuator: Actuator | None = None
    actuator_initial_condition: Any = None
    command_source: CommandSource | None = None
    input_builder: InputBuilder | None = None
