"""Tests for engine orchestration using a nonphysical local test double only."""

from __future__ import annotations

import numpy as np
import pytest

from gnc_lab.sim.engine import SimulationEngine
from gnc_lab.sim.types import SimulationConfig, SimulationMetadata, SimulationPlan


class ConstantRateTestDouble:
    """A minimal generic RHS used only to validate engine sequencing."""

    def default_input(self) -> float:
        """Provide a constant input without representing any physical system."""
        return 2.0

    def derivative(self, time: float, state: np.ndarray, plant_input: float) -> np.ndarray:
        """Return a constant derivative for an engine-level numerical check."""
        del time, state
        return np.array([plant_input])

    def output(self, time: float, state: np.ndarray, plant_input: float) -> dict[str, float]:
        """Return an arbitrary truth output suitable for recording checks."""
        del time, plant_input
        return {"value": float(state[0])}


def test_engine_runs_the_minimal_open_loop_plan() -> None:
    """A plan with only model, state, timing, and metadata is supported."""
    plan = SimulationPlan(
        model=ConstantRateTestDouble(),
        initial_state=np.array([1.0]),
        config=SimulationConfig(start_time=0.0, stop_time=0.5, plant_dt=0.1),
        metadata=SimulationMetadata(name="engine test", seed=7),
    )

    result = SimulationEngine().run(plan)

    np.testing.assert_allclose(result.times, np.linspace(0.0, 0.5, 6))
    assert result.states[-1, 0] == pytest.approx(2.0)
    assert result.records[-1].output == {"value": pytest.approx(2.0)}
