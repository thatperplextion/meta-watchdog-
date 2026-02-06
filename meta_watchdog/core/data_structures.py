"""
Data Structures and Schemas for Meta-Watchdog System

This module defines all the data classes used throughout the system,
ensuring type safety and clear contracts between components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray


class Severity(Enum):
    """Severity levels for alerts and issues."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def numeric_value(self) -> int:
        """Get numeric value for comparisons (1-4)."""
        return self.value


class CauseCategory(Enum):
    """Categories of root causes for model degradation."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_INSTABILITY = "feature_instability"
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence"
    DISTRIBUTION_SHIFT = "distribution_shift"
    OUTLIER_CONTAMINATION = "outlier_contamination"
    MISSING_DATA = "missing_data"
    FEATURE_CORRELATION_CHANGE = "feature_correlation_change"
    TARGET_SHIFT = "target_shift"
    ENVIRONMENTAL_CHANGE = "environmental_change"
    MODEL_STALENESS = "model_staleness"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Types of recommended actions."""
    RETRAIN_FULL = "retrain_full"
    RETRAIN_INCREMENTAL = "retrain_incremental"
    FEATURE_ADJUSTMENT = "feature_adjustment"
    FEATURE_REMOVAL = "feature_removal"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_REPLACEMENT = "model_replacement"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    DATA_QUALITY_CHECK = "data_quality_check"
    HUMAN_REVIEW = "human_review"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    MONITORING_INCREASE = "monitoring_increase"
    FALLBACK_ACTIVATION = "fallback_activation"
    NO_ACTION = "no_action"


class ScenarioType(Enum):
    """Types of counterfactual scenarios."""
    FEATURE_DRIFT = "feature_drift"
    NOISE_INJECTION = "noise_injection"
    TREND_SHIFT = "trend_shift"
    FEATURE_DECAY = "feature_decay"
    OUTLIER_STORM = "outlier_storm"
    MISSING_VALUES = "missing_values"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    SCALE_SHIFT = "scale_shift"
    CATEGORY_SHIFT = "category_shift"


@dataclass
class Prediction:
    """
    A single prediction with its associated metadata.
    
    This is the atomic unit of prediction in the system.
    """
    value: Union[float, int, str, NDArray[np.floating]]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    input_hash: Optional[str] = None
    prediction_id: Optional[str] = None
    
    # Optional ground truth (filled in later when available)
    actual_value: Optional[Union[float, int, str]] = None
    was_correct: Optional[bool] = None
    error: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class PredictionBatch:
    """
    A batch of predictions made together.
    
    Used for batch processing and analysis.
    """
    predictions: List[Prediction]
    batch_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Batch-level statistics
    mean_confidence: Optional[float] = None
    confidence_std: Optional[float] = None
    
    def __post_init__(self):
        """Compute batch statistics."""
        if self.predictions:
            confidences = [p.confidence for p in self.predictions]
            self.mean_confidence = float(np.mean(confidences))
            self.confidence_std = float(np.std(confidences))
    
    def __len__(self) -> int:
        return len(self.predictions)
    
    def __iter__(self):
        return iter(self.predictions)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a time window.
    
    Captures the essential metrics for monitoring model performance.
    """
    timestamp: datetime
    window_size: int  # Number of predictions in this window
    
    # Core metrics
    accuracy: Optional[float] = None  # For classification
    mean_error: Optional[float] = None  # MAE for regression
    rmse: Optional[float] = None  # RMSE for regression
    
    # Confidence metrics
    mean_confidence: float = 0.0
    confidence_std: float = 0.0
    
    # Calibration metrics
    confidence_accuracy_gap: float = 0.0  # Mean confidence - actual accuracy
    overconfident_ratio: float = 0.0  # % of wrong predictions with high confidence
    underconfident_ratio: float = 0.0  # % of correct predictions with low confidence
    
    # Stability metrics
    prediction_variance: float = 0.0
    confidence_variance: float = 0.0
    
    # Error distribution
    error_percentile_90: Optional[float] = None
    error_percentile_95: Optional[float] = None
    max_error: Optional[float] = None
    
    # Trend indicators
    accuracy_trend: float = 0.0  # Positive = improving, Negative = degrading
    confidence_trend: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityScore:
    """
    The Model Reliability Score (0-100).
    
    A composite score representing overall model health.
    """
    score: float  # 0-100
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Component scores (each 0-100)
    performance_score: float = 100.0
    calibration_score: float = 100.0  # Confidence calibration
    stability_score: float = 100.0
    freshness_score: float = 100.0  # Model not stale
    feature_health_score: float = 100.0
    
    # Component weights (should sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "performance": 0.30,
        "calibration": 0.25,
        "stability": 0.20,
        "freshness": 0.10,
        "feature_health": 0.15
    })
    
    # Status
    status: str = "healthy"  # healthy, warning, degraded, critical
    
    # Factors contributing to score reduction
    degradation_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and compute score."""
        if not 0.0 <= self.score <= 100.0:
            raise ValueError(f"Score must be in [0, 100], got {self.score}")
        self._update_status()
    
    def _update_status(self):
        """Update status based on score."""
        if self.score >= 80:
            self.status = "healthy"
        elif self.score >= 60:
            self.status = "warning"
        elif self.score >= 40:
            self.status = "degraded"
        else:
            self.status = "critical"
    
    @classmethod
    def compute(
        cls,
        performance_score: float,
        calibration_score: float,
        stability_score: float,
        freshness_score: float,
        feature_health_score: float,
        weights: Optional[Dict[str, float]] = None
    ) -> "ReliabilityScore":
        """
        Factory method to compute reliability score from components.
        
        Args:
            performance_score: Model performance (0-100)
            calibration_score: Confidence calibration (0-100)
            stability_score: Prediction stability (0-100)
            freshness_score: Model freshness (0-100)
            feature_health_score: Feature health (0-100)
            weights: Optional custom weights
            
        Returns:
            Computed ReliabilityScore
        """
        default_weights = {
            "performance": 0.30,
            "calibration": 0.25,
            "stability": 0.20,
            "freshness": 0.10,
            "feature_health": 0.15
        }
        w = weights or default_weights
        
        score = (
            w["performance"] * performance_score +
            w["calibration"] * calibration_score +
            w["stability"] * stability_score +
            w["freshness"] * freshness_score +
            w["feature_health"] * feature_health_score
        )
        
        # Identify degradation factors
        degradation_factors = []
        if performance_score < 70:
            degradation_factors.append("low_performance")
        if calibration_score < 70:
            degradation_factors.append("poor_calibration")
        if stability_score < 70:
            degradation_factors.append("instability")
        if freshness_score < 70:
            degradation_factors.append("model_staleness")
        if feature_health_score < 70:
            degradation_factors.append("feature_degradation")
        
        return cls(
            score=score,
            performance_score=performance_score,
            calibration_score=calibration_score,
            stability_score=stability_score,
            freshness_score=freshness_score,
            feature_health_score=feature_health_score,
            weights=w,
            degradation_factors=degradation_factors
        )


@dataclass
class FailurePrediction:
    """
    Prediction of future model failure.
    
    The core output of the meta-failure prediction model.
    """
    failure_probability: float  # 0-1
    time_to_failure: Optional[int]  # Estimated predictions until failure
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Confidence in this prediction (meta-meta level!)
    prediction_confidence: float = 0.5
    
    # Contributing factors
    contributing_factors: List[str] = field(default_factory=list)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    
    # Failure type prediction
    predicted_failure_type: Optional[str] = None
    
    # Trend information
    probability_trend: float = 0.0  # Positive = increasing risk
    
    # Urgency level
    urgency: Severity = Severity.LOW
    
    def __post_init__(self):
        """Validate and set urgency."""
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError(f"Failure probability must be in [0, 1]")
        self._compute_urgency()
    
    def _compute_urgency(self):
        """Compute urgency based on probability and time."""
        if self.failure_probability >= 0.8 or (
            self.time_to_failure is not None and self.time_to_failure < 10
        ):
            self.urgency = Severity.CRITICAL
        elif self.failure_probability >= 0.6 or (
            self.time_to_failure is not None and self.time_to_failure < 50
        ):
            self.urgency = Severity.HIGH
        elif self.failure_probability >= 0.4 or (
            self.time_to_failure is not None and self.time_to_failure < 100
        ):
            self.urgency = Severity.MEDIUM
        else:
            self.urgency = Severity.LOW


@dataclass
class RootCause:
    """
    An identified root cause of model degradation.
    """
    category: CauseCategory
    description: str
    severity: Severity
    confidence: float  # Confidence in this diagnosis (0-1)
    
    # Evidence supporting this cause
    evidence: List[str] = field(default_factory=list)
    evidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Affected components
    affected_features: List[str] = field(default_factory=list)
    
    # Timeline
    detected_at: datetime = field(default_factory=datetime.now)
    estimated_onset: Optional[datetime] = None
    
    # Recommended actions for this specific cause
    suggested_actions: List[ActionType] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate confidence."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1]")


@dataclass
class Recommendation:
    """
    An actionable recommendation.
    """
    action_type: ActionType
    title: str
    description: str
    priority: Severity
    
    # What triggers this recommendation
    triggered_by: List[str] = field(default_factory=list)  # Cause IDs or descriptions
    
    # Expected impact
    expected_improvement: float = 0.0  # Expected reliability score improvement
    confidence_in_recommendation: float = 0.5
    
    # Implementation details
    implementation_steps: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    requires_human: bool = False
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    """
    A human-readable explanation of system findings.
    """
    # The three key questions
    what_is_happening: str
    why_is_it_happening: str
    what_should_be_done: str
    
    # Full narrative
    full_explanation: str
    
    # Summary (one-liner)
    summary: str
    
    # Detail level
    detail_level: str = "standard"  # brief, standard, detailed
    
    # Technical details (for advanced users)
    technical_details: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence in explanation
    explanation_confidence: float = 0.8
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SensitivityResult:
    """
    Result of sensitivity analysis for a feature or scenario.
    """
    feature_name: str
    scenario_type: ScenarioType
    
    # Sensitivity scores
    sensitivity_score: float  # 0-1, higher = more sensitive
    impact_magnitude: float  # Expected performance drop
    
    # Threshold at which model breaks
    failure_threshold: Optional[float] = None
    
    # Confidence intervals
    sensitivity_lower: float = 0.0
    sensitivity_upper: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """
    Result of a counterfactual stress test.
    """
    scenario_type: ScenarioType
    severity: float  # Severity of the stress (0-1)
    
    # Performance under stress
    baseline_performance: float
    stressed_performance: float
    performance_drop: float
    
    # Model behavior
    predictions_changed_ratio: float
    confidence_change: float
    
    # Failure indicators
    failure_triggered: bool = False
    failure_point: Optional[float] = None  # Severity at which failure occurred
    
    # Sensitivity details
    most_affected_features: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemState:
    """
    Complete state of the Meta-Watchdog system at a point in time.
    
    This is the comprehensive snapshot used for system awareness.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Current health
    reliability_score: Optional[ReliabilityScore] = None
    failure_prediction: Optional[FailurePrediction] = None
    
    # Recent performance
    recent_metrics: Optional[PerformanceMetrics] = None
    
    # Identified issues
    root_causes: List[RootCause] = field(default_factory=list)
    
    # Current recommendations
    recommendations: List[Recommendation] = field(default_factory=list)
    
    # Latest explanation
    explanation: Optional[Explanation] = None
    
    # Stress test results
    stress_test_results: List[StressTestResult] = field(default_factory=list)
    
    # System metadata
    total_predictions: int = 0
    predictions_since_last_retrain: int = 0
    model_version: str = "unknown"
    
    # Alert status
    active_alerts: List[str] = field(default_factory=list)
    alert_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reliability_score": self.reliability_score.score if self.reliability_score else None,
            "reliability_status": self.reliability_score.status if self.reliability_score else "unknown",
            "failure_probability": self.failure_prediction.failure_probability if self.failure_prediction else None,
            "time_to_failure": self.failure_prediction.time_to_failure if self.failure_prediction else None,
            "root_cause_count": len(self.root_causes),
            "recommendation_count": len(self.recommendations),
            "total_predictions": self.total_predictions,
            "model_version": self.model_version,
            "alert_count": self.alert_count,
        }
