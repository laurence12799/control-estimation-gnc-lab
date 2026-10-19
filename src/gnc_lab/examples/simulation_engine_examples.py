"""Small test-only systems that exercise the generic simulation engine.

These are deliberately kept outside :mod:`gnc_lab.models`. They demonstrate
engine contracts and plotting, not project dynamics, controls, estimation, or
GNC implementations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

from gnc_lab.analysis.plotting import plot_state_comparison, plot_state_ensemble
from gnc_lab.sim.contracts import ContinuousModel
from gnc_lab.sim.engine import SimulationEngine
from gnc_lab.sim.randomness import NamedRandomStreams
from gnc_lab.sim.types import (
    Measurement,
    ModelSnapshot,
    PeriodicSchedule,
    SensorBinding,
    SimulationConfig,
    SimulationMetadata,
    SimulationPlan,
    SimulationResult,
    TimedComponent,
)


@dataclass(frozen=True)
class FirstOrderLagExample:
    """A scalar stable lag used only for simulation-engine verification."""

    time_constant: float
    input_value: float = 1.0

    def __post_init__(self) -> None:
        if self.time_constant <= 0.0:
            raise ValueError("time_constant must be positive.")

    def default_input(self) -> float:
        """Return the fixed open-loop forcing used by the example."""
        return self.input_value

    def derivative(self, time: float, state: np.ndarray, plant_input: float) -> np.ndarray:
        """Evaluate the scalar test equation."""
        del time
        return np.array([(float(plant_input) - state[0]) / self.time_constant])

    def output(self, time: float, state: np.ndarray, plant_input: float) -> dict[str, float]:
        """Expose a compact truth output for engine logging."""
        del time, plant_input
        return {"state": float(state[0])}


@dataclass(frozen=True)
class HarmonicOscillatorExample:
    """A free two-state oscillator used only for engine verification."""

    natural_frequency: float

    def __post_init__(self) -> None:
        if self.natural_frequency <= 0.0:
            raise ValueError("natural_frequency must be positive.")

    def default_input(self) -> float:
        """Return the zero forcing used by this free-response example."""
        return 0.0

    def derivative(self, time: float, state: np.ndarray, plant_input: float) -> np.ndarray:
        """Evaluate a generic unit-mass oscillator with optional forcing."""
        del time
        return np.array(
            [
                state[1],
                -(self.natural_frequency**2) * state[0] + float(plant_input),
            ]
        )

    def output(self, time: float, state: np.ndarray, plant_input: float) -> dict[str, float]:
        """Expose position and velocity for recorded-output coverage."""
        del time, plant_input
        return {"position": float(state[0]), "velocity": float(state[1])}


@dataclass(frozen=True)
class MassSpringDamperExample:
    """A forced second-order response used only for engine verification."""

    damping: float
    mass: float = 1.0
    spring: float = 1.0
    input_value: float = 1.0

    def __post_init__(self) -> None:
        if self.mass <= 0.0 or self.spring <= 0.0 or self.damping < 0.0:
            raise ValueError("mass and spring must be positive and damping non-negative.")

    def default_input(self) -> float:
        """Return the fixed step forcing used by the example."""
        return self.input_value

    def derivative(self, time: float, state: np.ndarray, plant_input: float) -> np.ndarray:
        """Evaluate the generic forced mass-spring-damper test equation."""
        del time
        acceleration = float(plant_input) - self.damping * state[1] - self.spring * state[0]
        return np.array([state[1], acceleration / self.mass])

    def output(self, time: float, state: np.ndarray, plant_input: float) -> dict[str, float]:
        """Expose displacement and velocity for recorded-output coverage."""
        del time, plant_input
        return {"position": float(state[0]), "velocity": float(state[1])}


@dataclass(frozen=True)
class PositionEstimateExample:
    """A deliberately small estimate payload for feedback-routing demonstration."""

    position: float


@dataclass
class PositionSensorExample:
    """A perfect test-only position sensor with observable sample count."""

    sample_count: int = 0

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Reset the sensor counter for one example run."""
        del initial_condition, rng
        self.sample_count = 0

    def sample(self, snapshot: ModelSnapshot, rng: Generator) -> float:
        """Return the first test-state element as a position measurement."""
        del rng
        self.sample_count += 1
        return float(snapshot.state[0])

    def snapshot(self) -> dict[str, int]:
        """Return the count needed to inspect multi-rate sampling history."""
        return {"samples": self.sample_count}


@dataclass
class MeasurementHoldEstimatorExample:
    """A test-only estimator that holds its most recent position measurement.

    It illustrates the estimator interface and sample-and-hold timing. It is
    not a reference observer or a recommended estimator design.
    """

    current_estimate: PositionEstimateExample = PositionEstimateExample(position=0.0)
    propagation_count: int = 0

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Initialize the held estimate from a scalar example condition."""
        del rng
        initial_position = 0.0 if initial_condition is None else float(initial_condition)
        self.current_estimate = PositionEstimateExample(position=initial_position)
        self.propagation_count = 0

    def propagate(self, start_time: float, duration: float, plant_input: float) -> None:
        """Record propagation events while intentionally retaining the last sample."""
        del start_time, duration, plant_input
        self.propagation_count += 1

    def update(self, time: float, measurements: dict[str, Measurement]) -> None:
        """Replace the estimate with the newest test-only position measurement."""
        del time
        self.current_estimate = PositionEstimateExample(
            position=float(measurements["position"].value)
        )

    def estimate(self) -> PositionEstimateExample:
        """Return the current immutable demonstration estimate."""
        return self.current_estimate

    def snapshot(self) -> dict[str, float | int]:
        """Expose sample-held state for post-processing inspection."""
        return {
            "position": self.current_estimate.position,
            "propagations": self.propagation_count,
        }


@dataclass
class PositionFeedbackExampleController:
    """A minimal proportional law used only to reveal feedback routing.

    This is intentionally not a reusable controller implementation. It exists
    solely to make truth feedback and a held estimate produce distinct engine
    traces under otherwise identical test conditions.
    """

    gain: float = 2.0
    setpoint: float = 1.0
    feedback_kinds: list[str] | None = None

    def reset(self, initial_condition: Any, rng: Generator) -> None:
        """Clear feedback-routing evidence for one example run."""
        del initial_condition, rng
        self.feedback_kinds = []

    def update(self, time: float, reference: Any, feedback: Any) -> float:
        """Return a test-only proportional command from the supplied feedback."""
        del time, reference
        if isinstance(feedback, ModelSnapshot):
            position = float(feedback.state[0])
            feedback_kind = "truth"
        elif isinstance(feedback, PositionEstimateExample):
            position = feedback.position
            feedback_kind = "estimate"
        else:
            raise TypeError(f"Unexpected feedback type: {type(feedback)!r}")
        if self.feedback_kinds is None:
            raise RuntimeError("Controller must be reset before update.")
        self.feedback_kinds.append(feedback_kind)
        return self.gain * (self.setpoint - position)

    def snapshot(self) -> dict[str, int | str | None]:
        """Expose the most recent feedback route for the result history."""
        last_kind = self.feedback_kinds[-1] if self.feedback_kinds else None
        return {"feedback_updates": len(self.feedback_kinds or []), "last_kind": last_kind}


def run_example_simulations() -> dict[str, dict[str, SimulationResult]]:
    """Run two open-loop cases for each test-only example system."""
    engine = SimulationEngine()
    cases: dict[str, dict[str, SimulationResult]] = {
        "first_order_lag": {
            "tau = 1.0 s": _run_open_loop(
                engine,
                "first-order lag, tau 1.0 s",
                FirstOrderLagExample(time_constant=1.0),
                np.array([0.0]),
                duration=4.0,
            ),
            "tau = 0.35 s": _run_open_loop(
                engine,
                "first-order lag, tau 0.35 s",
                FirstOrderLagExample(time_constant=0.35),
                np.array([0.0]),
                duration=4.0,
            ),
        },
        "harmonic_oscillator": {
            "omega = 1.0 rad/s": _run_open_loop(
                engine,
                "harmonic oscillator, omega 1.0 rad/s",
                HarmonicOscillatorExample(natural_frequency=1.0),
                np.array([1.0, 0.0]),
                duration=8.0,
            ),
            "omega = 1.8 rad/s": _run_open_loop(
                engine,
                "harmonic oscillator, omega 1.8 rad/s",
                HarmonicOscillatorExample(natural_frequency=1.8),
                np.array([1.0, 0.0]),
                duration=8.0,
            ),
        },
        "mass_spring_damper": {
            "damping = 0.2": _run_open_loop(
                engine,
                "mass-spring-damper, damping 0.2",
                MassSpringDamperExample(damping=0.2),
                np.array([0.0, 0.0]),
                duration=10.0,
            ),
            "damping = 1.0": _run_open_loop(
                engine,
                "mass-spring-damper, damping 1.0",
                MassSpringDamperExample(damping=1.0),
                np.array([0.0, 0.0]),
                duration=10.0,
            ),
        },
    }
    return cases


def run_oscillator_frequency_monte_carlo(
    seed: int = 20261018,
    trial_count: int = 24,
) -> tuple[np.ndarray, tuple[SimulationResult, ...]]:
    """Run a reproducible frequency-variation ensemble for the oscillator.

    The variations represent uncertain model parameters, not sensor or actuator
    noise. A named experiment-level random stream keeps the sampled parameter
    set reproducible and separate from component-local engine streams.
    """
    if trial_count <= 0:
        raise ValueError("trial_count must be positive.")

    variation_rng = NamedRandomStreams(seed).generator("example:oscillator-frequency")
    frequencies = np.clip(variation_rng.normal(loc=1.0, scale=0.12, size=trial_count), 0.1, None)
    engine = SimulationEngine()
    results = tuple(
        _run_open_loop(
            engine,
            name=f"oscillator frequency Monte Carlo trial {trial_index:02d}",
            model=HarmonicOscillatorExample(natural_frequency=float(frequency)),
            initial_state=np.array([1.0, 0.0]),
            duration=8.0,
            metadata=SimulationMetadata(
                name=f"oscillator frequency Monte Carlo trial {trial_index:02d}",
                seed=seed,
                tags=("simulation-engine-example", "monte-carlo"),
                attributes={
                    "natural_frequency_rad_s": f"{frequency:.17g}",
                    "trial": str(trial_index),
                },
            ),
        )
        for trial_index, frequency in enumerate(frequencies)
    )
    return frequencies, results


def run_truth_vs_estimator_feedback_example() -> dict[str, SimulationResult]:
    """Run equivalent test plants with truth and sample-held estimate feedback.

    The estimated-feedback case uses a perfect but slow position sensor and a
    deliberately simplistic measurement-hold estimator. This isolates feedback
    routing and multi-rate timing; it is not an estimator-performance example.
    """
    return {
        "Truth feedback": _run_feedback_case(
            name="truth feedback routing",
            with_estimator=False,
        ),
        "Held measurement estimate": _run_feedback_case(
            name="estimated feedback routing",
            with_estimator=True,
        ),
    }


def _run_feedback_case(name: str, with_estimator: bool) -> SimulationResult:
    controller = PositionFeedbackExampleController()
    estimator = MeasurementHoldEstimatorExample()
    sensor = PositionSensorExample()
    plan = SimulationPlan(
        model=MassSpringDamperExample(damping=0.4, input_value=0.0),
        initial_state=np.array([0.0, 0.0]),
        config=SimulationConfig(start_time=0.0, stop_time=8.0, plant_dt=0.01),
        metadata=SimulationMetadata(
            name=name,
            seed=20261018,
            tags=("simulation-engine-example", "feedback-routing"),
        ),
        controller=TimedComponent(controller, PeriodicSchedule(period=0.02)),
        estimator=estimator if with_estimator else None,
        estimator_initial_condition=0.0 if with_estimator else None,
        sensors=(SensorBinding("position", sensor, PeriodicSchedule(period=0.2)),)
        if with_estimator
        else (),
    )
    return SimulationEngine().run(plan)


def render_example_figures(output_directory: Path) -> dict[str, Path]:
    """Run all examples and save the documentation figures without a GUI."""
    cases = run_example_simulations()
    specifications = {
        "first_order_lag": ("First-order lag engine example", ("State",)),
        "harmonic_oscillator": (
            "Harmonic oscillator engine example",
            ("Position", "Velocity"),
        ),
        "mass_spring_damper": (
            "Mass-spring-damper engine example",
            ("Position", "Velocity"),
        ),
    }
    figures = {
        system_name: plot_state_comparison(
            system_cases,
            state_labels=specifications[system_name][1],
            title=specifications[system_name][0],
            output_path=output_directory / f"{system_name}.png",
        )
        for system_name, system_cases in cases.items()
    }
    feedback_cases = run_truth_vs_estimator_feedback_example()
    figures["truth_vs_estimator_feedback"] = plot_state_comparison(
        feedback_cases,
        state_labels=("Position", "Velocity"),
        title="Truth versus held-estimate feedback",
        output_path=output_directory / "truth_vs_estimator_feedback.png",
    )
    frequencies, monte_carlo_results = run_oscillator_frequency_monte_carlo()
    figures["oscillator_frequency_monte_carlo"] = plot_state_ensemble(
        monte_carlo_results,
        state_index=0,
        state_label="Position",
        title="Oscillator frequency Monte Carlo",
        output_path=output_directory / "oscillator_frequency_monte_carlo.png",
        parameter_values=frequencies,
        parameter_label="Natural frequency [rad/s]",
    )
    return figures


def _run_open_loop(
    engine: SimulationEngine,
    name: str,
    model: ContinuousModel,
    initial_state: np.ndarray,
    duration: float,
    metadata: SimulationMetadata | None = None,
) -> SimulationResult:
    plan = SimulationPlan(
        model=model,
        initial_state=initial_state,
        config=SimulationConfig(start_time=0.0, stop_time=duration, plant_dt=0.01),
        metadata=metadata or SimulationMetadata(name=name, tags=("simulation-engine-example",)),
    )
    return engine.run(plan)


def main() -> None:
    """Render the checked-in documentation figures from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("docs/figures/simulation-engine"),
        help="Directory to receive PNG figures.",
    )
    arguments = parser.parse_args()
    paths = render_example_figures(arguments.output_directory)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
