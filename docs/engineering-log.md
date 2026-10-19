# Engineering log

## 2026-10-18 - Repository foundation

**Decision.** Start with one public Python repository and a project-local,
locked environment managed by uv.

**Evidence.** The six-month plan shares conventions, validation practices, and
eventually host-to-embedded interfaces across the controls, estimation, GNC,
hardware, and HIL milestones.

**Scope boundary.** This initial scaffold contains no dynamics, control,
estimation, guidance, or simulation implementation.

**Verification.** `uv.lock` was generated with Python 3.12.7. The local
environment synchronized successfully; Ruff linting and formatting checks and
the neutral package test pass.

**Next action.** The project owner selects and implements the first engineering
artifact. Architectural support remains available without adding domain-specific
dynamics, control, estimation, or GNC code.

## 2026-10-18 - Simulation kernel

**Decision.** Use a fixed-step RK4 engine with zero-order-held plant input and
component schedules aligned to the plant step.

**Evidence.** The planned systems have different state and signal definitions,
but share state propagation, discrete update events, reproducible random
effects, and signal recording.

**Scope boundary.** The engine exposes contracts for models, sensors,
estimators, planners, controllers, actuators, and input builders. It implements
none of those domain components.

**Next action.** The project owner can introduce a first user-owned model and
its model-specific validation tests when ready.

## 2026-10-18 - Simulation-engine verification harness

**Decision.** Keep engine-validation doubles and three simple propagation
examples outside `gnc_lab.models`, while adding a stable state-history plotting
utility and result provenance for post-processing.

**Evidence.** The kernel must be verified across all optional-component
combinations, multi-rate schedules, zero-order hold, and isolated random
streams before it becomes shared infrastructure for user-owned systems.

**Scope boundary.** The first-order lag, harmonic oscillator, and
mass-spring-damper cases are test-only contract examples, not portfolio models
or proposed controller, estimator, planner, or GNC implementations.

**Verification.** The suite exercises 48 valid optional-component assemblies,
plus integration, scheduling, noise isolation, provenance, and plotting. The
three reproducible state-history figures are saved in
`docs/figures/simulation-engine/`.

## 2026-10-18 - Component contract guide

**Decision.** Document the simulation lifecycle and data ownership at the
component boundary before adding user-owned controls or estimation code.

**Evidence.** Correct controller and estimator interpretation depends on exact
sample/update timing, zero-order hold, feedback type, random-stream ownership,
and the distinction between command, actuation, and plant input.

**Scope boundary.** The guide specifies communication contracts only; it adds
no project model, controller, estimator, planner, or GNC implementation.

## 2026-10-18 - Monte Carlo and feedback-routing examples

**Decision.** Add test-only examples for model-parameter variation and the
truth-versus-estimator feedback boundary, alongside the original open-loop
propagation examples.

**Evidence.** Parameter uncertainty and a transition from truth to estimated
feedback are central early control-engineering workflows. The examples exercise
named experiment-level random streams, metadata provenance, multi-rate sensor
and controller timing, component snapshots, and post-processing plots.

**Scope boundary.** The Monte Carlo runner, held-measurement estimator, and
proportional feedback law are small teaching fixtures under `gnc_lab.examples`.
They are not project model, controller, or estimator implementations.

## 2026-10-18 - First Quarto report source

**Decision.** Establish a self-contained Quarto project for milestone reports,
beginning with the oscillator-frequency Monte Carlo infrastructure example.

**Evidence.** A report that binds a claim, assumptions, reproduction steps,
metrics, failure cases, and resulting actions to a versioned figure makes the
engineering record reviewable without turning routine notes into long reports.

**Scope boundary.** The source report is not a deployed site or a physical
system validation claim. GitHub Pages remains deferred until the first
user-owned engineering milestone is ready to publish.
