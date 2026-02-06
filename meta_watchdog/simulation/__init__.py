"""
Simulation module for counterfactual scenario testing.
"""

from meta_watchdog.simulation.scenarios import (
    ScenarioGenerator,
    ScenarioConfig,
)
from meta_watchdog.simulation.counterfactual import CounterfactualSimulator
from meta_watchdog.simulation.sensitivity import SensitivityMapper

__all__ = [
    "ScenarioGenerator",
    "ScenarioConfig",
    "CounterfactualSimulator",
    "SensitivityMapper",
]
