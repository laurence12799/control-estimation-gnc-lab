"""Contracts between the simulation kernel and future user-owned components."""

from __future__ import annotations

from typing import Any, Protocol

from numpy.random import Generator

from gnc_lab.core.types import FloatVector
from gnc_lab.sim.types import Measurement, ModelSnapshot


class ContinuousModel(Protocol):
    """A continuous-time plant with a numerically integrable state vector.

    Concrete models define the meaning and type of ``plant_input`` and
    ``output``. The engine does not impose a state naming convention or a
    physics-specific interface.
    """

    def default_input(self) -> Any:
        """Return the quiescent plant input for an uncontrolled simulation."""

    def derivative(self, time: float, state: FloatVector, plant_input: Any) -> FloatVector:
        """Evaluate the state derivative at one numerical integration stage."""

    def output(self, time: float, state: FloatVector, plant_input: Any) -> Any:
        """Return model-defined truth outputs for logging or sensing."""


class Planner(Protocol):
    """Produces a reference from the selected feedback representation."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Initialize internal state for one simulation run."""

    def update(self, time: float, feedback: Any) -> Any:
        """Return the reference held until the next scheduled update."""


class Controller(Protocol):
    """Produces a command from a reference and selected feedback."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Initialize internal state for one simulation run."""

    def update(self, time: float, reference: Any, feedback: Any) -> Any:
        """Return the command held until the next scheduled update."""


class Estimator(Protocol):
    """Maintains an estimated feedback state independently of the plant truth."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Initialize internal state for one simulation run."""

    def propagate(self, start_time: float, duration: float, plant_input: Any) -> None:
        """Advance the estimate across one plant integration interval."""

    def update(self, time: float, measurements: dict[str, Measurement]) -> None:
        """Incorporate the measurements sampled at ``time``."""

    def estimate(self) -> Any:
        """Return the feedback representation supplied to other components."""


class Sensor(Protocol):
    """Samples a model snapshot and may maintain sensor-specific internal state."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Initialize internal state for one simulation run."""

    def sample(self, snapshot: ModelSnapshot, rng: Generator) -> Any:
        """Return one model-specific measurement value."""


class Actuator(Protocol):
    """Maps a command to actuation at discrete command-update instants.

    Continuous actuator physics should instead be represented as plant state or
    as a future composite plant model, so it is integrated with the plant.
    """

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Initialize internal state for one simulation run."""

    def apply(self, time: float, command: Any, rng: Generator) -> Any:
        """Return actuation for the interval beginning at ``time``."""


class CommandSource(Protocol):
    """Produces open-loop commands when no controller is configured."""

    def command(self, time: float, feedback: Any) -> Any:
        """Return the command for the interval beginning at ``time``."""


class InputBuilder(Protocol):
    """Combines actuation and an optional environment into a model input."""

    def build(self, time: float, actuation: Any, snapshot: ModelSnapshot) -> Any:
        """Return the model-defined plant input for the next integration interval."""


class ComponentStateProvider(Protocol):
    """Optional protocol for components that expose recordable internal state."""

    def snapshot(self) -> Any:
        """Return a serializable state snapshot for run provenance."""
