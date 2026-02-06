"""
Core module containing base classes, interfaces, and data structures.
"""

from meta_watchdog.core.interfaces import (
    PredictionModel,
    Monitor,
    Analyzer,
    Recommender,
    Explainer,
)
from meta_watchdog.core.base_model import BaseModel, ModelType
from meta_watchdog.core.data_structures import (
    Prediction,
    PredictionBatch,
    PerformanceMetrics,
    ReliabilityScore,
    FailurePrediction,
    RootCause,
    Recommendation,
    Explanation,
    SystemState,
)

__all__ = [
    # Interfaces
    "PredictionModel",
    "Monitor", 
    "Analyzer",
    "Recommender",
    "Explainer",
    # Base Model
    "BaseModel",
    "ModelType",
    # Data Structures
    "Prediction",
    "PredictionBatch",
    "PerformanceMetrics",
    "ReliabilityScore",
    "FailurePrediction",
    "RootCause",
    "Recommendation",
    "Explanation",
    "SystemState",
]
