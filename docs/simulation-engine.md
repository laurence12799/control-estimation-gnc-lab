# Simulation engine

## Purpose

`gnc_lab.sim` provides reusable orchestration for systems that share a
continuous state-propagation loop but differ in state definitions, inputs,
outputs, and algorithm choices. It contains no physical model, controller,
estimator, planner, or guidance implementation.

For the implementation-facing data and lifecycle contract for every optional
component, see [simulation component contracts](component-contracts.md).

## Minimal open-loop simulation

A user-owned model supplies a finite one-dimensional initial state,
`default_input()`, `derivative()`, and `output()`. A `SimulationPlan` then needs
only that model, the initial state, fixed-step timing, and metadata. The engine
propagates the model under its default input and records truth history.

## Closed-loop assembly

Future simulations can add components incrementally:

1. Use truth feedback with no estimator for early control experiments.
2. Add sensors to produce timestamped imperfect measurements.
3. Add an estimator, which becomes the feedback source for the planner and
   controller.
4. Add a planner and controller at their own periodic update rates.
5. Add a discrete actuator and an input builder for model-specific environment
   or disturbance composition.

The same resolved `SimulationConfig`, `SimulationMetadata`, component schedule,
and named random streams should be used for A/B comparisons. Use stable sensor
names across compared runs when paired noise realizations are desired.

`SimulationResult` retains the resolved configuration and metadata, copied
initial conditions, and one `StepRecord` at every plant boundary. Each record
contains truth state and output; command, actuation, and plant input; feedback
and reference; only the sensor measurements sampled at that boundary; and
optional component-state snapshots. A component may expose `snapshot()` to make
its own serializable internal state available for post-processing. This remains
optional so the kernel does not prescribe algorithm state representations.

## Component timing

All version-1 component schedules must be integer multiples of `plant_dt`.
At each integration boundary, due sensor samples are collected first, then the
estimator incorporates them, followed by planner and controller updates. The
plant is then propagated one `plant_dt` interval with its input held constant.
At the initial boundary, actuation is applied once after the scheduled updates;
the engine does not consume a redundant actuator/noise sample before a
time-zero controller update.

An estimator's `propagate(start_time, duration, plant_input)` method is called
after the plant advances, before measurements at the next boundary are used.
This supports the standard predict-then-update lifecycle without imposing an
estimation algorithm.

## Current boundaries

- RK4 is the default fixed-step integrator.
- Continuous actuator dynamics are model state, not an actuator plug-in.
- The engine records in memory. Add streaming/columnar output only when Monte
  Carlo result size demonstrates the need.
- Model authors define state names, input types, output types, units, frames,
  sensors' truth quantities, and validation tests.

## Test-only engine examples

`gnc_lab.examples.simulation_engine_examples` keeps its examples outside
`gnc_lab.models`, so none is presented as a project model or reusable algorithm
implementation. It currently demonstrates:

- Two open-loop cases each for a first-order lag, harmonic oscillator, and
  mass-spring-damper response.
- An oscillator-frequency Monte Carlo experiment. A named experiment-level RNG
  stream samples each natural frequency; every trial stores the exact sampled
  value in `SimulationMetadata.attributes`, and the figure shows the ensemble,
  percentile envelope, and parameter distribution.
- Identical mass-spring-damper test plants using truth feedback and a perfect
  but slow, sample-held position estimate. The intentionally simplistic
  estimator and proportional controller exist only to make feedback routing and
  multi-rate timing observable; neither is a recommended engineering design.

Run `uv run python -m gnc_lab.examples.simulation_engine_examples` to render
all figures.
