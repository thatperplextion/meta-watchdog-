"""
Recommendations module for action generation.

This module provides tools for generating actionable recommendations
based on model health analysis.
"""

from meta_watchdog.recommendations.action_engine import (
    ActionRecommendationEngine,
    ActionRecommendation,
    ActionPlan,
    ActionPriority,
    ActionCategory,
)

__all__ = [
    "ActionRecommendationEngine",
    "ActionRecommendation",
    "ActionPlan",
    "ActionPriority",
    "ActionCategory",
]
