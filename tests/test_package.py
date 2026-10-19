"""Project-level checks that do not depend on any engineering implementation."""

import gnc_lab


def test_package_version_is_defined() -> None:
    """The package exposes a version for diagnostics and release provenance."""
    assert gnc_lab.__version__ == "0.1.0"
