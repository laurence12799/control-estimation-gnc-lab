"""Tests for numerical infrastructure, not any engineering plant model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gnc_lab.sim.integrators import RK4Integrator


def test_rk4_matches_exponential_solution_for_generic_rhs() -> None:
    """Classical RK4 has the expected accuracy on a nonphysical scalar RHS."""
    integrator = RK4Integrator()
    initial_state = np.array([1.0])

    result = integrator.step(
        lambda time, state, plant_input: state,
        time=0.0,
        state=initial_state,
        duration=0.1,
        plant_input=None,
    )

    assert result[0] == pytest.approx(math.exp(0.1), abs=1e-6)
