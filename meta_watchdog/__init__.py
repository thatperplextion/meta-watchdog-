"""
Meta-Watchdog: Self-Aware Machine Learning System

A comprehensive failure-anticipating ML system that:
- Continuously monitors model performance and confidence
- Predicts when and why failures will occur
- Simulates counterfactual "what-if" scenarios  
- Provides human-readable explanations and recommendations

This is not just drift detection - it's Self-Aware AI.
"""

__version__ = "1.0.0"
__author__ = "Meta-Watchdog Team"

# Core components
from meta_watchdog.core.base_model import BaseModel
from meta_watchdog.core.data_structures import (
    Prediction,
    PredictionBatch,
    PerformanceMetrics,
    ReliabilityScore,
    FailurePrediction,
)

# Orchestrator - the main entry point
from meta_watchdog.orchestrator.system import (
    MetaWatchdogOrchestrator,
    OrchestratorConfig,
)

# Monitoring
from meta_watchdog.monitoring.performance_monitor import PerformanceMonitor
from meta_watchdog.monitoring.reliability_scorer import ReliabilityScoringEngine

# Meta-prediction
from meta_watchdog.meta_prediction.failure_predictor import MetaFailurePredictor

# Simulation
from meta_watchdog.simulation.counterfactual import CounterfactualSimulator
from meta_watchdog.simulation.sensitivity import SensitivityMapper

# Analysis
from meta_watchdog.analysis.root_cause import RootCauseAnalyzer

# Recommendations
from meta_watchdog.recommendations.action_engine import ActionRecommendationEngine

# Explainability
from meta_watchdog.explainability.explanation_engine import ExplainabilityEngine

# Dashboard
from meta_watchdog.dashboard.data_provider import DashboardDataProvider
from meta_watchdog.dashboard.terminal import TerminalDashboard

# Testing utilities
from meta_watchdog.testing.synthetic_data import SyntheticDataGenerator

__all__ = [
    # Version
    "__version__",
    
    # Core
    "BaseModel",
    "Prediction",
    "PredictionBatch",
    "PerformanceMetrics",
    "ReliabilityScore",
    "FailurePrediction",
    
    # Main orchestrator
    "MetaWatchdogOrchestrator",
    "OrchestratorConfig",
    
    # Monitoring
    "PerformanceMonitor",
    "ReliabilityScoringEngine",
    
    # Meta-prediction
    "MetaFailurePredictor",
    
    # Simulation
    "CounterfactualSimulator",
    "SensitivityMapper",
    
    # Analysis
    "RootCauseAnalyzer",
    
    # Recommendations
    "ActionRecommendationEngine",
    
    # Explainability
    "ExplainabilityEngine",
    
    # Dashboard
    "DashboardDataProvider",
    "TerminalDashboard",
    
    # Testing
    "SyntheticDataGenerator",
]
