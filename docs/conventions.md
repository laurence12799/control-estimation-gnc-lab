# Engineering conventions

## Repository baseline

- Python: 3.12, recorded in `.python-version`.
- Dependencies: declared in `pyproject.toml`; exact resolution recorded in
  `uv.lock`.
- Generated experiment output: `runs/`, excluded from version control.
- Source of truth: reviewed Python modules and named configurations, not
  notebooks or manually edited figures.
- Python code and documentation: follow the project
  [Python style policy](python-style.md), including Doxygen-style docstrings
  for public and private code.

## Conventions to establish before the first model

The first model review will explicitly define:

1. SI base units and any allowed derived units.
2. Reference frames, handedness, axis directions, and gravity sign.
3. State-vector ordering and control-input ordering.
4. Time representation, integrator assumptions, and sample-time behavior.
5. Random-number-generator ownership and seed recording.
6. Plot labels, uncertainty notation, and numerical tolerances.

No convention is assumed silently. Each model and experiment must reference the
relevant documented choices.

## Simulation timing

- `plant_dt` is the fixed plant-integration step.
- Version-1 sensor, planner, and controller schedules are integer multiples of
  `plant_dt` and use a phase relative to simulation start time.
- The engine holds the plant input during each RK4 integration interval.
- An estimator propagates during the interval and updates when a due sensor
  measurement arrives at the next integration boundary.
- The run seed is split into stable, name-derived component streams. Record
  stable component names whenever a random realization matters.
