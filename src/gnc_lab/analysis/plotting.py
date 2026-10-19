"""Reproducible non-interactive plots for recorded simulation histories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gnc_lab.sim.types import SimulationResult


def plot_state_comparison(
    cases: Mapping[str, SimulationResult],
    state_labels: Sequence[str],
    title: str,
    output_path: Path,
) -> Path:
    """Save one state-history figure comparing one or more simulation cases."""
    if not cases:
        raise ValueError("At least one simulation case is required for a plot.")

    results = tuple(cases.items())
    state_dimension = results[0][1].states.shape[1]
    if len(state_labels) != state_dimension:
        raise ValueError("state_labels must contain one label for each state element.")
    if any(result.states.shape[1] != state_dimension for _, result in results):
        raise ValueError("Every compared result must have the same state dimension.")

    figure, axes_grid = plt.subplots(
        nrows=state_dimension,
        ncols=1,
        sharex=True,
        figsize=(8.0, 2.6 * state_dimension),
        layout="constrained",
    )
    axes = tuple(axes_grid if state_dimension > 1 else (axes_grid,))

    for state_index, axis in enumerate(axes):
        for case_name, result in results:
            axis.plot(result.times, result.states[:, state_index], label=case_name)
        axis.set_ylabel(state_labels[state_index])
        axis.grid(visible=True, alpha=0.3)

    axes[0].set_title(title)
    axes[0].legend()
    axes[-1].set_xlabel("Time [s]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_state_ensemble(
    results: Sequence[SimulationResult],
    state_index: int,
    state_label: str,
    title: str,
    output_path: Path,
    parameter_values: np.ndarray,
    parameter_label: str,
) -> Path:
    """Save a Monte-Carlo state ensemble with percentile envelope and sample histogram."""
    if not results:
        raise ValueError("At least one simulation result is required for an ensemble plot.")
    if len(results) != len(parameter_values):
        raise ValueError("parameter_values must contain one entry per simulation result.")

    times = results[0].times
    trajectories = []
    for result in results:
        if result.states.shape[1] <= state_index:
            raise ValueError("state_index is outside a result's state dimension.")
        if not np.array_equal(result.times, times):
            raise ValueError("Every ensemble result must use the same time grid.")
        trajectories.append(result.states[:, state_index])
    values = np.stack(trajectories)

    figure, (history_axis, histogram_axis) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.0, 4.4),
        layout="constrained",
        width_ratios=(2.4, 1.0),
    )
    for trajectory in values:
        history_axis.plot(times, trajectory, color="C0", alpha=0.18, linewidth=0.8)
    lower, median, upper = np.quantile(values, (0.05, 0.5, 0.95), axis=0)
    history_axis.fill_between(times, lower, upper, color="C0", alpha=0.25, label="5–95%")
    history_axis.plot(times, median, color="C0", linewidth=2.0, label="Median")
    history_axis.set_title(title)
    history_axis.set_xlabel("Time [s]")
    history_axis.set_ylabel(state_label)
    history_axis.grid(visible=True, alpha=0.3)
    history_axis.legend()

    histogram_axis.hist(parameter_values, bins="auto", color="C1", edgecolor="white")
    histogram_axis.set_xlabel(parameter_label)
    histogram_axis.set_ylabel("Trials")
    histogram_axis.grid(visible=True, axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
