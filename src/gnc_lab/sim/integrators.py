"""Numerical integration methods independent of any specific plant model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from gnc_lab.core.types import FloatVector, as_float_vector

StateDerivative = Callable[[float, FloatVector, Any], FloatVector]


class Integrator(Protocol):
    """Advances a continuous state over one fixed, zero-order-held interval."""

    def step(
        self,
        derivative: StateDerivative,
        time: float,
        state: FloatVector,
        duration: float,
        plant_input: Any,
    ) -> FloatVector:
        """Return the state at ``time + duration``."""


class RK4Integrator:
    """Classical fourth-order Runge-Kutta with a held plant input."""

    def step(
        self,
        derivative: StateDerivative,
        time: float,
        state: FloatVector,
        duration: float,
        plant_input: Any,
    ) -> FloatVector:
        """Advance one fixed integration interval.

        The control/plant input is intentionally held for all four stages. The
        simulation engine updates commanded actuation only at its configured
        discrete event times.
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
