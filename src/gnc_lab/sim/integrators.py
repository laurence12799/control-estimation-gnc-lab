"""@file integrators.py
@brief Provide plant-agnostic fixed-step numerical integration methods.

@details Integrators receive a model derivative callable and make no
assumptions about a model's engineering domain or input representation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from gnc_lab.core.types import FloatVector, as_float_vector

StateDerivative = Callable[[float, FloatVector, Any], FloatVector]


class Integrator(Protocol):
    """@brief Define the interface for one fixed, zero-order-held state step.

    Concrete implementations may use any appropriate numerical scheme, but
    must leave plant-input interpretation to the supplied derivative.
    """

    def step(
        self,
        derivative: StateDerivative,
        time: float,
        state: FloatVector,
        duration: float,
        plant_input: Any,
    ) -> FloatVector:
        """@brief Advance a continuous state by one fixed interval.

        @param derivative StateDerivative: Right-hand-side function to
            evaluate.
        @param time float: Start time of the integration interval [s].
        @param state FloatVector: State at ``time``.
        @param duration float: Positive interval duration [s].
        @param plant_input Any: Input held throughout the interval.
        @return FloatVector: State at ``time + duration``.
        """


class RK4Integrator:
    """@brief Implement classical fourth-order Runge-Kutta (RK4).

    @note The implementation passes an unchanged plant input to all four
    derivative evaluations, preserving the engine's zero-order-hold contract.
    """

    def step(
        self,
        derivative: StateDerivative,
        time: float,
        state: FloatVector,
        duration: float,
        plant_input: Any,
    ) -> FloatVector:
        """@brief Advance a state through one RK4 integration interval.

        @param derivative StateDerivative: Right-hand-side function to
            evaluate at RK4 stages.
        @param time float: Start time of the interval [s].
        @param state FloatVector: State at the start of the interval.
        @param duration float: Positive RK4 step size [s].
        @param plant_input Any: Input held unchanged across all RK4 stages.
        @return FloatVector: Validated state at the end of the interval.
        @raises ValueError: If ``duration`` is non-positive, a derivative is
            invalid, or a derivative shape differs from the state shape.
        """
        if duration <= 0.0:
            raise ValueError("Integration duration must be positive.")

        initial_state = as_float_vector(state, name="state")
        k1 = self._evaluate(derivative, time, initial_state, plant_input)
        k2 = self._evaluate(
            derivative,
            time + duration / 2.0,
            initial_state + duration * k1 / 2.0,
            plant_input,
        )
        k3 = self._evaluate(
            derivative,
            time + duration / 2.0,
            initial_state + duration * k2 / 2.0,
            plant_input,
        )
        k4 = self._evaluate(
            derivative,
            time + duration,
            initial_state + duration * k3,
            plant_input,
        )
        return as_float_vector(
            initial_state + duration * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
            name="integrated state",
        )

    @staticmethod
    def _evaluate(
        derivative: StateDerivative,
        time: float,
        state: FloatVector,
        plant_input: Any,
    ) -> FloatVector:
        """@brief Evaluate and validate one RK4 derivative stage.

        @param derivative StateDerivative: Model right-hand-side callable.
        @param time float: Stage time [s].
        @param state FloatVector: State supplied to the stage.
        @param plant_input Any: Zero-order-held plant input.
        @return FloatVector: Finite derivative matching ``state.shape``.
        @raises ValueError: If the derivative is invalid or has the wrong
            shape.
        @note This private helper passes a copied state to prevent a model from
            mutating the integrator's stage state.
        """
        value = as_float_vector(
            derivative(time, state.copy(), plant_input),
            name="state derivative",
        )
        if value.shape != state.shape:
            raise ValueError(
                "State derivative shape must match the state shape; "
                f"received {value.shape} for state shape {state.shape}."
            )
        return value
