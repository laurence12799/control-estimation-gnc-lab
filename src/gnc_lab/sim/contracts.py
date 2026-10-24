"""@file contracts.py
@brief Declare contracts between the simulation kernel and user components.

@details These protocols specify communication and lifecycle behavior without
providing any model, controller, estimator, planner, or GNC implementation.
"""

from __future__ import annotations

from typing import Any, Protocol

from numpy.random import Generator

from gnc_lab.core.types import FloatVector
from gnc_lab.sim.types import Measurement, ModelSnapshot


class ContinuousModel(Protocol):
    """@brief Define a continuous-time plant with an integrable state vector.

    Concrete models define the meaning and type of ``plant_input`` and
    ``output``. The engine does not impose a state naming convention or a
    physics-specific interface.
    """

    def default_input(self) -> Any:
        """@brief Return the quiescent input for an uncontrolled simulation.

        @return Any: Model-defined default plant input.
        """

    def derivative(
        self, time: float, state: FloatVector, plant_input: Any
    ) -> FloatVector:
        """@brief Evaluate the state derivative at one integration stage.

        @param time float: Stage time [s].
        @param state FloatVector: Plant state at ``time``.
        @param plant_input Any: Input held for the plant interval.
        @return FloatVector: Finite derivative matching ``state.shape``.
        """

    def output(self, time: float, state: FloatVector, plant_input: Any) -> Any:
        """@brief Return model-defined truth outputs for logging or sensing.

        @param time float: Output evaluation time [s].
        @param state FloatVector: Truth state at ``time``.
        @param plant_input Any: Input used for the output calculation.
        @return Any: Model-defined output payload.
        """


class Planner(Protocol):
    """@brief Define a component that produces references from feedback."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """@brief Initialize planner state for one simulation run.

        @param initial_condition Any: User-supplied initialization payload.
        @param rng Generator: Planner-specific reproducible random stream.
        """

    def update(self, time: float, feedback: Any) -> Any:
        """@brief Produce a reference to hold until the next planner update.

        @param time float: Planner update time [s].
        @param feedback Any: Truth snapshot or estimator output.
        @return Any: Model- and controller-defined reference payload.
        """


class Controller(Protocol):
    """@brief Define a component that produces commands from feedback."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """@brief Initialize controller state for one simulation run.

        @param initial_condition Any: User-supplied initialization payload.
        @param rng Generator: Controller-specific reproducible random stream.
        """

    def update(self, time: float, reference: Any, feedback: Any) -> Any:
        """@brief Produce a command to hold until the next controller update.

        @param time float: Controller update time [s].
        @param reference Any: Most recent planner output, if any.
        @param feedback Any: Truth snapshot or estimator output.
        @return Any: Model- and actuator-defined command payload.
        """


class Estimator(Protocol):
    """@brief Define an estimator independent of plant-truth feedback."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """@brief Initialize estimator state for one simulation run.

        @param initial_condition Any: User-supplied initialization payload.
        @param rng Generator: Estimator-specific reproducible random stream.
        """

    def propagate(self, start_time: float, duration: float, plant_input: Any) -> None:
        """@brief Advance the estimate across one plant integration interval.

        @param start_time float: Interval start time [s].
        @param duration float: Propagation duration [s].
        @param plant_input Any: Zero-order-held plant input for the interval.
        """

    def update(self, time: float, measurements: dict[str, Measurement]) -> None:
        """@brief Incorporate measurements sampled at one boundary.

        @param time float: Measurement time [s].
        @param measurements dict[str, Measurement]: Measurements keyed by
            sensor name.
        """

    def estimate(self) -> Any:
        """@brief Return the feedback representation for downstream components.

        @return Any: Estimator-defined state or output estimate.
        """


class Sensor(Protocol):
    """@brief Define a component that samples a plant-truth snapshot."""

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """@brief Initialize sensor state for one simulation run.

        @param initial_condition Any: User-supplied initialization payload.
        @param rng Generator: Sensor-specific reproducible random stream.
        """

    def sample(self, snapshot: ModelSnapshot, rng: Generator) -> Any:
        """@brief Produce one model-specific measurement value.

        @param snapshot ModelSnapshot: Truth data at the sample time.
        @param rng Generator: Sensor-specific reproducible random stream.
        @return Any: Sensor-defined measurement payload.
        """


class Actuator(Protocol):
    """@brief Define a discrete component that maps commands to actuation.

    Continuous actuator physics should instead be represented as plant state or
    as a future composite plant model, so it is integrated with the plant.
    """

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """@brief Initialize actuator state for one simulation run.

        @param initial_condition Any: User-supplied initialization payload.
        @param rng Generator: Actuator-specific reproducible random stream.
        """

    def apply(self, time: float, command: Any, rng: Generator) -> Any:
        """@brief Map a command to actuation for the next plant interval.

        @param time float: Command-update time [s].
        @param command Any: Controller or command-source output.
        @param rng Generator: Actuator-specific reproducible random stream.
        @return Any: Actuation held until the next command update.
        """


class CommandSource(Protocol):
    """@brief Define an open-loop command producer without a controller."""

    def command(self, time: float, feedback: Any) -> Any:
        """@brief Produce a command for the interval beginning at ``time``.

        @param time float: Command evaluation time [s].
        @param feedback Any: Truth snapshot or estimator output.
        @return Any: Command payload for the next plant interval.
        """


class InputBuilder(Protocol):
    """@brief Define a mapper from actuation to a model-specific plant input."""

    def build(self, time: float, actuation: Any, snapshot: ModelSnapshot) -> Any:
        """@brief Build the input for the next plant integration interval.

        @param time float: Boundary time [s].
        @param actuation Any: Most recently applied actuator output.
        @param snapshot ModelSnapshot: Current plant-truth data.
        @return Any: Model-defined plant input held through the interval.
        """


class ComponentStateProvider(Protocol):
    """@brief Define an optional interface for recordable component state."""

    def snapshot(self) -> Any:
        """@brief Return a serializable snapshot for run provenance.

        @return Any: Component-defined state payload safe to record.
        """
