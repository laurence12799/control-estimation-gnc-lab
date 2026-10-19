"""Tests for fixed-step multi-rate scheduling infrastructure."""

import pytest

from gnc_lab.sim.types import PeriodicSchedule


def test_schedule_is_due_at_aligned_steps() -> None:
    """A two-step schedule triggers at zero and every second plant step."""
    schedule = PeriodicSchedule(period=0.02)

    assert schedule.is_due(0, plant_dt=0.01)
    assert not schedule.is_due(1, plant_dt=0.01)
    assert schedule.is_due(2, plant_dt=0.01)


def test_schedule_rejects_an_unaligned_period() -> None:
    """Version 1 avoids ambiguous event timing by enforcing grid alignment."""
    schedule = PeriodicSchedule(period=0.015)

    with pytest.raises(ValueError, match="integer multiple"):
        schedule.is_due(0, plant_dt=0.01)
