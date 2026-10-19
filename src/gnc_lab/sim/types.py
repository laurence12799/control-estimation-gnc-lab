"""Data structures for simulation timing, assembly, and recorded results."""

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
    """Human and reproducibility metadata attached to every simulation result."""

    name: str
    seed: int = 0
    revision: str | None = None
    tags: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Simulation metadata requires a non-empty name.")
        if self.seed < 0:
            raise ValueError("Simulation metadata seed must be non-negative.")


@dataclass(frozen=True)
class SimulationConfig:
    """Fixed-step timing for a deterministic numerical experiment."""

    start_time: float
    stop_time: float
    plant_dt: float

    def __post_init__(self) -> None:
        if self.plant_dt <= 0.0:
            raise ValueError("plant_dt must be positive.")
        if self.stop_time <= self.start_time:
            raise ValueError("stop_time must be greater than start_time.")
        self.number_of_steps()

    def number_of_steps(self) -> int:
        """Return the exact number of plant steps, rejecting a partial final step."""
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
        """Return the simulation time at a zero-based integration boundary."""
        return self.start_time + step_index * self.plant_dt


@dataclass(frozen=True)
class PeriodicSchedule:
    """A fixed-rate update schedule aligned to the plant time grid."""

    period: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.period <= 0.0:
            raise ValueError("Schedule period must be positive.")
        if self.phase < 0.0 or self.phase >= self.period:
            raise ValueError(
                "Schedule phase must be greater than or equal to zero and below period."
            )

    def is_due(self, step_index: int, plant_dt: float) -> bool:
        """Return whether this schedule is due at the specified plant step."""
        period_steps = self._aligned_steps(self.period, plant_dt, "period")
        phase_steps = self._aligned_steps(self.phase, plant_dt, "phase")
        return step_index >= phase_steps and (step_index - phase_steps) % period_steps == 0

    @staticmethod
    def _aligned_steps(value: float, plant_dt: float, label: str) -> int:
        raw_steps = value / plant_dt
        steps = round(raw_steps)
        tolerance = max(1e-12, abs(raw_steps) * 1e-12)
        if not np.isclose(raw_steps, steps, rtol=0.0, atol=tolerance):
            raise ValueError(f"Schedule {label} must be an integer multiple of plant_dt.")
        return int(steps)


@dataclass(frozen=True)
class TimedComponent[ComponentT]:
    """Binds an optional component to a schedule and initial condition."""

    component: ComponentT
    schedule: PeriodicSchedule
    initial_condition: Any = None


@dataclass(frozen=True)
class SensorBinding:
    """Names one sensor and defines when it samples the plant truth."""

    name: str
    sensor: Sensor
    schedule: PeriodicSchedule
    initial_condition: Any = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Sensor bindings require a non-empty name.")


@dataclass(frozen=True)
class ModelSnapshot:
    """Truth data visible to sensor and input-builder infrastructure."""

    time: float
    state: FloatVector
    plant_input: Any
    output: Any


@dataclass(frozen=True)
class Measurement:
    """A timestamped sensor value produced during a simulation event."""

    sensor_name: str
    time: float
    value: Any


@dataclass(frozen=True)
class StepRecord:
    """One integration-boundary record with truth and signal provenance."""

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
    """Copied initial conditions retained with a completed simulation result."""

    plant_state: FloatVector
    command: Any
    planner: Any
    controller: Any
    estimator: Any
    actuator: Any
    sensors: Mapping[str, Any]


@dataclass(frozen=True)
class SimulationResult:
    """In-memory run result; large-scale storage is a later concern."""

    config: SimulationConfig
    metadata: SimulationMetadata
    initial_conditions: SimulationInitialConditions
    records: tuple[StepRecord, ...]

    @property
    def times(self) -> np.ndarray:
        """Return the recorded integration-boundary timestamps."""
        return np.asarray([record.time for record in self.records], dtype=np.float64)

    @property
    def states(self) -> np.ndarray:
        """Return the truth-state history with one row per recorded timestamp."""
        return np.stack([record.state for record in self.records])


@dataclass(frozen=True)
class SimulationPlan:
    """Assembles user-owned components around the reusable simulation kernel."""

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
