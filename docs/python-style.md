# Python code and documentation style

This repository follows its standard Python style guide as the coding and
documentation baseline. It supplements, but does not replace, the engineering
conventions in
[conventions.md](conventions.md).

## Intent

The project will keep both public interfaces and private implementation helpers
readable to a control/GNC engineer returning to an experiment months later. New
or modified Python code should follow this policy:

- Use typed, PEP 8-compatible code with explicit imports, clear names, and
  small single-purpose functions.
- Use a module docstring beginning with `@file` and `@brief` to state each
  module's responsibility.
- Use class docstrings beginning with `@brief`. Dataclass docstrings document
  their fields with `@param`; ordinary class docstrings describe the purpose,
  retained state, and lifecycle where relevant.
- Document every public and private function or method. Begin with `@brief`,
  list inputs with `@param`, outputs with `@return`, expected errors with
  `@raises`, and important usage constraints with `@note` or `@warning`.
  Tags that do not apply, such as `@return` for a procedure returning `None`,
  are omitted.
- Use unit suffixes in variable and parameter names when units are not already
  clear from an established data contract. Simulation time values are seconds
  unless documented otherwise.
- Keep external algorithms and all user-authored engineering models separate
  from reusable simulation infrastructure. Test doubles and examples must say
  explicitly that they are not engineering implementations.

## Enforcement and review

Ruff checks documentation coverage for public modules, classes, methods,
functions, and magic methods. The test suite checks docstring presence across
all project Python definitions, including private helpers. The detailed
semantic review remains part of code review: a docstring must accurately
describe the actual data contract and timing behavior, not merely satisfy a
presence check.

Run the same checks locally before committing:

```powershell
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

The guide is a style baseline, not a source of engineering requirements.
Repository documentation, tests, and explicit component contracts remain the
source of truth for simulation behavior.
