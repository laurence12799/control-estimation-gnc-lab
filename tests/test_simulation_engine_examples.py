"""Tests for the test-only examples that demonstrate engine use and plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gnc_lab.examples.simulation_engine_examples import (
    PositionEstimateExample,
    render_example_figures,
    run_example_simulations,
    run_oscillator_frequency_monte_carlo,
    run_truth_vs_estimator_feedback_example,
)
from gnc_lab.sim.types import ModelSnapshot


def test_each_engine_example_runs_two_finite_open_loop_cases() -> None:
    """The examples provide two runnable cases per simple dynamic-system type."""
    cases = run_example_simulations()

    assert set(cases) == {
        "first_order_lag",
        "harmonic_oscillator",
        "mass_spring_damper",
    }
    for system_cases in cases.values():
        assert len(system_cases) == 2
        for result in system_cases.values():
            assert result.metadata.tags == ("simulation-engine-example",)
            assert np.all(np.isfinite(result.states))
            assert result.times[0] == 0.0
            assert result.times[-1] > result.times[0]


def test_engine_examples_render_state_history_figures(tmp_path: Path) -> None:
    """State-dynamics figures are generated without a GUI or notebook session."""
    figures = render_example_figures(tmp_path)

    assert set(figures) == {
        "first_order_lag",
        "harmonic_oscillator",
        "mass_spring_damper",
        "oscillator_frequency_monte_carlo",
        "truth_vs_estimator_feedback",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures.values())


def test_oscillator_frequency_monte_carlo_is_reproducible_and_records_variations() -> (
    None
):
    """Parameter variation has a stable trial set and retains each sampled value."""
    first_frequencies, first_results = run_oscillator_frequency_monte_carlo(
        seed=123, trial_count=8
    )
    second_frequencies, second_results = run_oscillator_frequency_monte_carlo(
        seed=123, trial_count=8
    )

    np.testing.assert_array_equal(first_frequencies, second_frequencies)
    assert np.std(first_frequencies) > 0.0
    for trial_index, (first, second) in enumerate(
        zip(first_results, second_results, strict=True)
    ):
        assert first.metadata.tags == ("simulation-engine-example", "monte-carlo")
        assert first.metadata.attributes["trial"] == str(trial_index)
        assert (
            float(first.metadata.attributes["natural_frequency_rad_s"])
            == first_frequencies[trial_index]
        )
        np.testing.assert_allclose(first.states, second.states)


def test_truth_and_estimator_feedback_examples_follow_distinct_data_routes() -> None:
    """The demonstration confirms feedback routing and multi-rate held estimates."""
    cases = run_truth_vs_estimator_feedback_example()
    truth = cases["Truth feedback"]
    estimate = cases["Held measurement estimate"]

    assert all(isinstance(record.feedback, ModelSnapshot) for record in truth.records)
    assert all(
        isinstance(record.feedback, PositionEstimateExample)
        for record in estimate.records
    )
    assert truth.records[-1].component_states["controller"]["last_kind"] == "truth"
    assert (
        estimate.records[-1].component_states["controller"]["last_kind"] == "estimate"
    )
    assert estimate.records[-1].component_states["sensors"]["position"]["samples"] == 41
    assert not np.allclose(truth.states, estimate.states)
