"""@file __init__.py
@brief Re-export reusable numerical simulation infrastructure.

No model, controller, estimator, or guidance implementation is provided here.
Those components are supplied by the project owner through the published
contracts.
"""

from gnc_lab.sim.engine import SimulationEngine
from gnc_lab.sim.integrators import RK4Integrator
from gnc_lab.sim.types import (
    PeriodicSchedule,
    SensorBinding,
    SimulationConfig,
    SimulationInitialConditions,
    SimulationMetadata,
    SimulationPlan,
    SimulationResult,
    TimedComponent,
)

__all__ = [
    "PeriodicSchedule",
    "RK4Integrator",
    "SensorBinding",
    "SimulationConfig",
    "SimulationEngine",
    "SimulationInitialConditions",
    "SimulationMetadata",
    "SimulationPlan",
    "SimulationResult",
    "TimedComponent",
]
