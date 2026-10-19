"""Tests for reproducible, named infrastructure random streams."""

import numpy as np

from gnc_lab.sim.randomness import NamedRandomStreams


def test_named_streams_are_reproducible_and_independent_of_request_order() -> None:
    """Component names, not creation order, determine their random realization."""
    first = NamedRandomStreams(123)
    first_sensor = first.generator("sensor:imu").normal(size=4)
    first.generator("sensor:barometer").normal(size=4)

    second = NamedRandomStreams(123)
    second.generator("sensor:barometer").normal(size=4)
    second_sensor = second.generator("sensor:imu").normal(size=4)

    np.testing.assert_allclose(first_sensor, second_sensor)
