"""
Meta-Prediction module for failure forecasting.
"""

from meta_watchdog.meta_prediction.failure_predictor import MetaFailurePredictor
from meta_watchdog.meta_prediction.trend_analyzer import TrendAnalyzer

__all__ = [
    "MetaFailurePredictor",
    "TrendAnalyzer",
]
