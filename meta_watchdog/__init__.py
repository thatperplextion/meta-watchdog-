"""
Meta-Watchdog: Self-Aware Machine Learning System

A failure-anticipating ML system that predicts its own reliability degradation
and provides actionable recommendations for safer AI decisions.
"""

__version__ = "0.1.0"
__author__ = "Meta-Watchdog Team"

from meta_watchdog.orchestrator.system import MetaWatchdogSystem

__all__ = ["MetaWatchdogSystem", "__version__"]
