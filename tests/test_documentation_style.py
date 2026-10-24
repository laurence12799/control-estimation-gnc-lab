"""Enforcement tests for the repository Python documentation policy."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_PYTHON_DIRECTORIES = (_PROJECT_ROOT / "src", _PROJECT_ROOT / "tests")


def test_every_project_definition_has_a_docstring() -> None:
    """@brief Require docstrings for modules and every class or function.

    @note This includes private helpers so later infrastructure maintenance has
    the same documentation baseline as the public API.
    """
    undocumented = [
        description
        for path in _project_python_files()
        for description in _undocumented_definitions(path)
    ]

    assert not undocumented, "Undocumented Python definitions:\n" + "\n".join(
        undocumented
    )


def _project_python_files() -> Iterator[Path]:
    """@brief Yield every repository-owned Python source and test file.

    @return Iterator[Path]: Deterministically ordered project Python paths.
    """
    for directory in _PYTHON_DIRECTORIES:
        yield from sorted(directory.rglob("*.py"))


def _undocumented_definitions(path: Path) -> Iterator[str]:
    """@brief Yield descriptions of undocumented definitions in one file.

    @param path Path: Repository-owned Python file to inspect.
    @return Iterator[str]: ``path:line:definition`` descriptions missing a
        docstring.
    """
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if ast.get_docstring(module) is None:
        yield f"{path.relative_to(_PROJECT_ROOT)}:1:module"

    for node in ast.walk(module):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is None:
                yield f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}:{node.name}"
