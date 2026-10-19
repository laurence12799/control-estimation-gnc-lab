# Architecture

## Current scope

This document defines boundaries for the repository skeleton. It does not
define or implement a project dynamics model, controller, estimator, guidance
law, or a specific engineering simulation experiment.

## Package boundaries

```text
src/gnc_lab/
  core/      Shared types, time, units, frames, and reproducibility helpers
  models/    Individually documented plant models
  sim/       Numerical integration, timing, component contracts, and provenance
  analysis/  Metrics, post-processing, and reproducible figures
```

Future work may add estimation, control, guidance, firmware, and hardware
directories when their interfaces and validation plans are explicitly agreed.

## Simulation kernel

The simulation engine is a fixed-step, multi-rate numerical kernel. It owns
time, numerical integration, component update order, random-number streams, and
recording. It does not own any physics or algorithm implementation.

```text
due sensors -> estimator update -> planner update -> controller update
     -> actuator -> input builder -> model output/log -> RK4 plant propagation
```

At the next integration boundary, an estimator first propagates over the
previous zero-order-held plant-input interval, then receives newly sampled
measurements. If no estimator is configured, the controller/planner receive an
explicit truth snapshot. This makes perfect-state-feedback experiments possible
without silently treating truth as an estimate.

### Interfaces

- `ContinuousModel` supplies `default_input`, a continuous state derivative,
  and model-defined truth output.
- `Sensor` samples a truth snapshot at its own fixed-rate schedule.
- `Estimator`, `Planner`, and `Controller` are optional scheduled components.
- `Actuator` maps a command to discrete actuation. Continuous actuator dynamics
  belong in a model state or future composite plant so they are integrated with
  the plant.
- `InputBuilder` combines actuation with model-specific exogenous effects such
  as environment or disturbances.

The kernel uses `numpy.ndarray` only for the integrated state. Inputs, outputs,
references, estimates, and measurements remain model-owned types. This keeps
cart-pole, motor, and aircraft interfaces free to differ without weakening the
numerical contract.

The exact lifecycle, data ownership, timing, and payload rules for component
authors are documented in [simulation component contracts](component-contracts.md).

### Numerical scope

Version 1 provides fixed-step classical RK4 with zero-order-held plant input.
Every periodic component schedule must be an integer multiple of `plant_dt`.
This is intentional: it gives deterministic timing and clear event ordering for
the early projects. Adaptive integration, root-event detection, transport delay,
and streamed high-volume storage are deferred extensions.

## Design rules

- A model, algorithm, and experiment runner have separate responsibilities.
- A named configuration and seed identify each repeatable experiment.
- Named component RNG streams derive from the run seed, so adding an unrelated
  sensor does not change an existing sensor's noise realization.
- Results retain copied setup data and boundary-by-boundary signals. Components
  may opt into internal-state provenance by exposing a serializable
  `snapshot()` method; the kernel does not define their internal state.
- Generated outputs belong in `runs/` and are not committed by default.
- Selected figures and reports are committed only when they support a reviewed
  technical claim.
- Notebooks may explore package APIs but do not become the canonical algorithm
  implementation.

## Dependency policy

Python packages are declared in `pyproject.toml` and resolved into `uv.lock`.
The lockfile is committed after local environment creation. Add a dependency
only when a reviewed milestone needs it.
