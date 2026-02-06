"""
Explainability module for human-readable explanations.

This module provides tools for generating clear, audience-appropriate
explanations of model health, failures, and recommendations.
"""

from meta_watchdog.explainability.explanation_engine import (
    ExplainabilityEngine,
    ExplanationAudience,
    ExplanationVerbosity,
    ExplanationSection,
    StructuredExplanation,
)

__all__ = [
    "ExplainabilityEngine",
    "ExplanationAudience",
    "ExplanationVerbosity",
    "ExplanationSection",
    "StructuredExplanation",
]
