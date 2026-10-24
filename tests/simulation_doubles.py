"""Nonphysical test doubles for exercising generic simulation infrastructure."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import Generator

from gnc_lab.sim.types import Measurement, ModelSnapshot


@dataclass
class ScalarInputTestDouble:
    """A generic scalar RHS that records every integration-stage input."""

    default_value: float = 0.0
    rhs_inputs: list[float] = field(default_factory=list)

    def default_input(self) -> float:
        """Return the scalar default input."""
        return self.default_value

    def derivative(
        self, time: float, state: np.ndarray, plant_input: float
    ) -> np.ndarray:
        """Record the held input and return a constant-rate generic RHS."""
        del time, state
        self.rhs_inputs.append(float(plant_input))
        return np.array([float(plant_input)])

    def output(
        self, time: float, state: np.ndarray, plant_input: float
    ) -> dict[str, float]:
        """Return a minimal truth output for orchestration tests."""
        del time, plant_input
        return {"value": float(state[0])}


class StaticTestDouble:
    """A zero-derivative generic plant for noise-isolation tests."""

    def default_input(self) -> float:
        """Return the quiescent scalar input."""
        return 0.0

    def derivative(
        self, time: float, state: np.ndarray, plant_input: float
    ) -> np.ndarray:
        """Return zero state change regardless of the input."""
        del time, plant_input
        return np.zeros_like(state)

    def output(
        self, time: float, state: np.ndarray, plant_input: float
    ) -> dict[str, float]:
        """Return the constant truth quantity sensed in noise tests."""
        del time, plant_input
        return {"value": float(state[0])}


@dataclass
class StatefulPlanner:
    """A scheduled reference source with inspectable state."""

    update_times: list[float] = field(default_factory=list)
    feedback_history: list[Any] = field(default_factory=list)
    initial_condition: Any = None

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Reset the local log for a new run."""
        del rng
        self.update_times = []
        self.feedback_history = []
        self.initial_condition = copy.deepcopy(initial_condition)

    def update(self, time: float, feedback: Any) -> float:
        """Record the event and issue a scalar reference."""
        self.update_times.append(time)
        self.feedback_history.append(feedback)
        return 0.5 + len(self.update_times)

    def snapshot(self) -> dict[str, Any]:
        """Expose serializable state for provenance assertions."""
        return {
            "initial_condition": self.initial_condition,
            "updates": len(self.update_times),
        }


@dataclass
class StatefulController:
    """A scheduled command source with inspectable feedback history."""

    update_times: list[float] = field(default_factory=list)
    feedback_history: list[Any] = field(default_factory=list)
    initial_condition: Any = None

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Reset local state for a new run."""
        del rng
        self.update_times = []
        self.feedback_history = []
        self.initial_condition = copy.deepcopy(initial_condition)

    def update(self, time: float, reference: Any, feedback: Any) -> float:
        """Record update inputs and issue a deterministic scalar command."""
        del reference
        self.update_times.append(time)
        self.feedback_history.append(feedback)
        return float(len(self.update_times))

    def snapshot(self) -> dict[str, Any]:
        """Expose serializable state for provenance assertions."""
        return {
            "initial_condition": self.initial_condition,
            "updates": len(self.update_times),
        }


@dataclass
class StatefulEstimator:
    """A minimal estimator-shaped component for kernel sequencing tests."""

    propagation_count: int = 0
    update_count: int = 0
    initial_condition: Any = None

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Reset counters for a new run."""
        del rng
        self.propagation_count = 0
        self.update_count = 0
        self.initial_condition = copy.deepcopy(initial_condition)

    def propagate(self, start_time: float, duration: float, plant_input: Any) -> None:
        """Record each plant-boundary propagation event."""
        del start_time, duration, plant_input
        self.propagation_count += 1

    def update(self, time: float, measurements: dict[str, Measurement]) -> None:
        """Record each measurement update event."""
        del time, measurements
        self.update_count += 1

    def estimate(self) -> dict[str, int]:
        """Return a distinct non-truth feedback representation."""
        return {"propagations": self.propagation_count, "updates": self.update_count}

    def snapshot(self) -> dict[str, Any]:
        """Expose serializable state for provenance assertions."""
        return {
            "initial_condition": self.initial_condition,
            "propagations": self.propagation_count,
            "updates": self.update_count,
        }


@dataclass
class NoiseSensor:
    """A sensor-shaped component with independent normally distributed noise."""

    noise_standard_deviation: float = 0.0
    sample_count: int = 0
    initial_condition: Any = None

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Reset measurement count for a new run."""
        del rng
        self.sample_count = 0
        self.initial_condition = copy.deepcopy(initial_condition)

    def sample(self, snapshot: ModelSnapshot, rng: Generator) -> float:
        """Return the truth value with reproducible sensor-local noise."""
        self.sample_count += 1
        return float(
            snapshot.state[0] + rng.normal(scale=self.noise_standard_deviation)
        )

    def snapshot(self) -> dict[str, Any]:
        """Expose serializable state for provenance assertions."""
        return {
            "initial_condition": self.initial_condition,
            "samples": self.sample_count,
        }


@dataclass
class BiasActuator:
    """A discrete actuator-shaped component with a fixed scalar bias."""

    bias: float = 0.25
    application_times: list[float] = field(default_factory=list)
    initial_condition: Any = None

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Reset application history for a new run."""
        del rng
        self.application_times = []
        self.initial_condition = copy.deepcopy(initial_condition)

    def apply(self, time: float, command: float, rng: Generator) -> float:
        """Record one discrete application and add the configured bias."""
        del rng
        self.application_times.append(time)
        return float(command) + self.bias

    def snapshot(self) -> dict[str, Any]:
        """Expose serializable state for provenance assertions."""
        return {
            "applications": len(self.application_times),
            "initial_condition": self.initial_condition,
        }


@dataclass
class RandomActuator:
    """An actuator double that consumes only its named random stream."""

    standard_deviation: float = 1.0

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """No internal state is needed for this double."""
        del initial_condition, rng

    def apply(self, time: float, command: float, rng: Generator) -> float:
        """Return a noisy actuation from the actuator-specific stream."""
        del time
        return float(command) + float(rng.normal(scale=self.standard_deviation))


@dataclass
class StatefulCommandSource:
    """An open-loop command producer for scheduling tests."""

    call_times: list[float] = field(default_factory=list)

    def command(self, time: float, feedback: Any) -> float:
        """Record each call and return a time-varying scalar command."""
        del feedback
        self.call_times.append(time)
        return 0.75 + time
