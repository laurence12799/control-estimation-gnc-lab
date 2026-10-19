"""Engine-level tests for isolated, reproducible component noise streams."""

from __future__ import annotations

import numpy as np

from gnc_lab.sim.engine import SimulationEngine
from gnc_lab.sim.types import (
    PeriodicSchedule,
    SensorBinding,
    SimulationConfig,
    SimulationMetadata,
    SimulationPlan,
    TimedComponent,
)
from tests.simulation_doubles import (
    NoiseSensor,
    RandomActuator,
    StatefulController,
    StaticTestDouble,
)


def test_sensor_noise_is_reproducible_for_the_same_run_seed() -> None:
    """Repeating a plan seed repeats the full primary-sensor noise sequence."""
    first = _primary_measurements(_noise_result(seed=41, include_unrelated_components=False))
    second = _primary_measurements(_noise_result(seed=41, include_unrelated_components=False))

    np.testing.assert_array_equal(first, second)


def test_primary_sensor_noise_is_isolated_from_unrelated_components() -> None:
    """Adding sensors and a noisy actuator does not advance the primary stream."""
    baseline = _primary_measurements(_noise_result(seed=41, include_unrelated_components=False))
    expanded = _primary_measurements(_noise_result(seed=41, include_unrelated_components=True))

    np.testing.assert_array_equal(baseline, expanded)


def test_sensor_noise_changes_when_the_run_seed_changes() -> None:
    """The run metadata seed actually selects a different random realization."""
    first = _primary_measurements(_noise_result(seed=41, include_unrelated_components=False))
    second = _primary_measurements(_noise_result(seed=42, include_unrelated_components=False))

    assert not np.array_equal(first, second)


def _noise_result(seed: int, include_unrelated_components: bool):
    sensors = [
        SensorBinding("primary", NoiseSensor(noise_standard_deviation=1.0), PeriodicSchedule(0.1))
    ]
    if include_unrelated_components:
        sensors.append(
            SensorBinding(
                "unrelated", NoiseSensor(noise_standard_deviation=1.0), PeriodicSchedule(0.1)
            )
        )

    return SimulationEngine().run(
        SimulationPlan(
            model=StaticTestDouble(),
            initial_state=np.array([0.0]),
            config=SimulationConfig(start_time=0.0, stop_time=0.4, plant_dt=0.1),
            metadata=SimulationMetadata(name="noise isolation", seed=seed),
            sensors=tuple(sensors),
            controller=(
                TimedComponent(StatefulController(), PeriodicSchedule(0.1))
                if include_unrelated_components
                else None
            ),
            actuator=RandomActuator() if include_unrelated_components else None,
        )
    )


def _primary_measurements(result) -> np.ndarray:
    return np.asarray(
        [record.measurements["primary"].value for record in result.records],
        dtype=np.float64,
    )
