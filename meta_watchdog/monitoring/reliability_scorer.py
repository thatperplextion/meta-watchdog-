"""
Reliability & Health Scoring Engine

Computes the Model Reliability Score (0-100) based on multiple
health indicators, providing a unified view of model trustworthiness.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from numpy.typing import NDArray

from meta_watchdog.core.data_structures import (
    ReliabilityScore,
    PerformanceMetrics,
    Severity,
)
from meta_watchdog.monitoring.metrics import RollingWindow


@dataclass
class HealthComponent:
    """A component of the overall health score."""
    name: str
    score: float  # 0-100
    weight: float  # Contribution weight
    status: str  # healthy, warning, critical
    details: Dict[str, Any] = field(default_factory=dict)
    
    def weighted_contribution(self) -> float:
        """Get weighted contribution to overall score."""
        return self.score * self.weight


class ReliabilityScoringEngine:
    """
    Computes and tracks the Model Reliability Score.
    
    The Reliability Score is a composite metric (0-100) that represents
    overall model health and trustworthiness. It considers:
    
    1. Performance Score: How well the model performs its task
    2. Calibration Score: How well confidence matches accuracy
    3. Stability Score: How stable predictions are over time
    4. Freshness Score: How recent the model training is
    5. Feature Health Score: Health of input features
    
    The score degrades PROACTIVELY as warning signs appear,
    not just when failure occurs.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        history_size: int = 100,
        degradation_sensitivity: float = 1.0,
    ):
        """
        Initialize the scoring engine.
        
        Args:
            weights: Custom weights for score components
            history_size: Size of score history to maintain
            degradation_sensitivity: How quickly score degrades (1.0 = normal)
        """
        self.weights = weights or {
            "performance": 0.30,
            "calibration": 0.25,
            "stability": 0.20,
            "freshness": 0.10,
            "feature_health": 0.15,
        }
        
        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        self.history_size = history_size
        self.degradation_sensitivity = degradation_sensitivity
        
        # Score history
        self._score_history: deque = deque(maxlen=history_size)
        self._component_history: Dict[str, RollingWindow] = {
            name: RollingWindow(history_size) for name in self.weights
        }
        
        # Current state
        self._current_score: Optional[ReliabilityScore] = None
        self._components: Dict[str, HealthComponent] = {}
        
        # Thresholds
        self._thresholds = {
            "healthy": 80,
            "warning": 60,
            "degraded": 40,
            "critical": 0,
        }
        
        # Degradation tracking
        self._degradation_factors: List[str] = []
        self._last_update = datetime.now()
    
    # ========== Score Computation ==========
    
    def compute_score(
        self,
        performance_metrics: Optional[PerformanceMetrics] = None,
        feature_stats: Optional[Dict[str, Any]] = None,
        model_age_predictions: int = 0,
        model_age_days: float = 0.0,
        failure_prediction: Optional[Dict[str, Any]] = None,
    ) -> ReliabilityScore:
        """
        Compute the current reliability score.
        
        Args:
            performance_metrics: Current performance metrics
            feature_stats: Feature health statistics
            model_age_predictions: Predictions since last training
            model_age_days: Days since last training
            failure_prediction: Failure prediction data
            
        Returns:
            Computed ReliabilityScore
        """
        self._degradation_factors = []
        
        # Compute individual component scores
        performance_score = self._compute_performance_score(performance_metrics)
        calibration_score = self._compute_calibration_score(performance_metrics)
        stability_score = self._compute_stability_score(performance_metrics)
        freshness_score = self._compute_freshness_score(
            model_age_predictions, model_age_days
        )
        feature_health_score = self._compute_feature_health_score(feature_stats)
        
        # Apply failure prediction penalty
        if failure_prediction:
            failure_penalty = self._compute_failure_penalty(failure_prediction)
            # Reduce all scores proportionally
            penalty_factor = 1 - (failure_penalty * 0.3)  # Max 30% reduction
            performance_score *= penalty_factor
            calibration_score *= penalty_factor
            stability_score *= penalty_factor
        
        # Store components
        self._components = {
            "performance": HealthComponent(
                name="performance",
                score=performance_score,
                weight=self.weights["performance"],
                status=self._get_component_status(performance_score),
            ),
            "calibration": HealthComponent(
                name="calibration",
                score=calibration_score,
                weight=self.weights["calibration"],
                status=self._get_component_status(calibration_score),
            ),
            "stability": HealthComponent(
                name="stability",
                score=stability_score,
                weight=self.weights["stability"],
                status=self._get_component_status(stability_score),
            ),
            "freshness": HealthComponent(
                name="freshness",
                score=freshness_score,
                weight=self.weights["freshness"],
                status=self._get_component_status(freshness_score),
            ),
            "feature_health": HealthComponent(
                name="feature_health",
                score=feature_health_score,
                weight=self.weights["feature_health"],
                status=self._get_component_status(feature_health_score),
            ),
        }
        
        # Compute weighted score
        reliability_score = ReliabilityScore.compute(
            performance_score=performance_score,
            calibration_score=calibration_score,
            stability_score=stability_score,
            freshness_score=freshness_score,
            feature_health_score=feature_health_score,
            weights=self.weights,
        )
        reliability_score.degradation_factors = self._degradation_factors.copy()
        
        # Update history
        self._score_history.append(reliability_score)
        for name, component in self._components.items():
            self._component_history[name].add(component.score)
        
        self._current_score = reliability_score
        self._last_update = datetime.now()
        
        return reliability_score
    
    def _compute_performance_score(
        self, metrics: Optional[PerformanceMetrics]
    ) -> float:
        """
        Compute performance component score.
        
        Based on accuracy (classification) or error metrics (regression).
        """
        if metrics is None:
            return 100.0  # Default to healthy if no data
        
        score = 100.0
        
        if metrics.accuracy is not None:
            # For classification: accuracy directly maps to score
            acc_contribution = metrics.accuracy * 100
            
            # Apply degradation sensitivity
            if metrics.accuracy < 0.8:
                penalty = (0.8 - metrics.accuracy) * 100 * self.degradation_sensitivity
                acc_contribution = max(0, acc_contribution - penalty)
                self._degradation_factors.append("low_accuracy")
            
            score = acc_contribution
        
        elif metrics.mean_error is not None:
            # For regression: lower error = higher score
            # Normalize error (assuming reasonable error ranges)
            # This needs domain-specific normalization in practice
            normalized_error = min(1.0, metrics.mean_error / 10.0)  # Placeholder normalization
            score = (1 - normalized_error) * 100
            
            if metrics.mean_error > 1.0:
                self._degradation_factors.append("high_error")
        
        # Check trend
        if metrics.accuracy_trend < -0.01:
            trend_penalty = abs(metrics.accuracy_trend) * 50 * self.degradation_sensitivity
            score = max(0, score - trend_penalty)
            self._degradation_factors.append("declining_performance")
        
        return max(0.0, min(100.0, score))
    
    def _compute_calibration_score(
        self, metrics: Optional[PerformanceMetrics]
    ) -> float:
        """
        Compute calibration component score.
        
        Based on confidence-accuracy alignment.
        """
        if metrics is None:
            return 100.0
        
        score = 100.0
        
        # Confidence-accuracy gap
        gap = abs(metrics.confidence_accuracy_gap)
        if gap > 0.1:
            gap_penalty = gap * 100 * self.degradation_sensitivity
            score -= gap_penalty
            self._degradation_factors.append("confidence_miscalibration")
        
        # Overconfidence penalty (more severe)
        if metrics.overconfident_ratio > 0.1:
            overconf_penalty = metrics.overconfident_ratio * 150 * self.degradation_sensitivity
            score -= overconf_penalty
            self._degradation_factors.append("overconfidence")
        
        # Underconfidence penalty (less severe)
        if metrics.underconfident_ratio > 0.2:
            underconf_penalty = metrics.underconfident_ratio * 50 * self.degradation_sensitivity
            score -= underconf_penalty
            self._degradation_factors.append("underconfidence")
        
        return max(0.0, min(100.0, score))
    
    def _compute_stability_score(
        self, metrics: Optional[PerformanceMetrics]
    ) -> float:
        """
        Compute stability component score.
        
        Based on prediction and confidence variance.
        """
        if metrics is None:
            return 100.0
        
        score = 100.0
        
        # High prediction variance indicates instability
        if metrics.prediction_variance > 0.1:
            var_penalty = metrics.prediction_variance * 100 * self.degradation_sensitivity
            score -= var_penalty
            self._degradation_factors.append("prediction_instability")
        
        # High confidence variance indicates uncertainty
        if metrics.confidence_variance > 0.05:
            conf_var_penalty = metrics.confidence_variance * 150 * self.degradation_sensitivity
            score -= conf_var_penalty
            self._degradation_factors.append("confidence_instability")
        
        return max(0.0, min(100.0, score))
    
    def _compute_freshness_score(
        self, 
        model_age_predictions: int,
        model_age_days: float
    ) -> float:
        """
        Compute freshness component score.
        
        Models degrade over time and usage.
        """
        score = 100.0
        
        # Prediction-based aging
        # Assume model should be retrained every ~10000 predictions
        prediction_staleness = min(1.0, model_age_predictions / 10000)
        if prediction_staleness > 0.5:
            pred_penalty = prediction_staleness * 50 * self.degradation_sensitivity
            score -= pred_penalty
            if prediction_staleness > 0.8:
                self._degradation_factors.append("model_staleness_predictions")
        
        # Time-based aging
        # Assume model should be reviewed every ~30 days
        time_staleness = min(1.0, model_age_days / 30)
        if time_staleness > 0.5:
            time_penalty = time_staleness * 30 * self.degradation_sensitivity
            score -= time_penalty
            if time_staleness > 0.8:
                self._degradation_factors.append("model_staleness_time")
        
        return max(0.0, min(100.0, score))
    
    def _compute_feature_health_score(
        self, feature_stats: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute feature health component score.
        
        Based on feature distribution stability and quality.
        """
        if feature_stats is None:
            return 100.0
        
        score = 100.0
        
        # Check for drift indicators
        drift_score = feature_stats.get("drift_score", 0.0)
        if drift_score > 0.1:
            drift_penalty = drift_score * 100 * self.degradation_sensitivity
            score -= drift_penalty
            self._degradation_factors.append("feature_drift")
        
        # Check for missing data
        missing_ratio = feature_stats.get("missing_ratio", 0.0)
        if missing_ratio > 0.05:
            missing_penalty = missing_ratio * 100 * self.degradation_sensitivity
            score -= missing_penalty
            self._degradation_factors.append("missing_data")
        
        # Check for outliers
        outlier_ratio = feature_stats.get("outlier_ratio", 0.0)
        if outlier_ratio > 0.05:
            outlier_penalty = outlier_ratio * 80 * self.degradation_sensitivity
            score -= outlier_penalty
            self._degradation_factors.append("outliers")
        
        # Check for correlation changes
        correlation_change = feature_stats.get("correlation_change", 0.0)
        if correlation_change > 0.2:
            corr_penalty = correlation_change * 50 * self.degradation_sensitivity
            score -= corr_penalty
            self._degradation_factors.append("correlation_shift")
        
        return max(0.0, min(100.0, score))
    
    def _compute_failure_penalty(
        self, failure_prediction: Dict[str, Any]
    ) -> float:
        """
        Compute penalty based on failure prediction.
        
        Higher failure probability = higher penalty.
        """
        prob = failure_prediction.get("failure_probability", 0.0)
        
        # Non-linear penalty: accelerates as probability increases
        penalty = prob ** 1.5
        
        # Time-to-failure urgency
        ttf = failure_prediction.get("time_to_failure")
        if ttf is not None and ttf < 100:
            urgency_factor = 1 + (1 - ttf / 100)
            penalty *= urgency_factor
        
        return min(1.0, penalty)
    
    def _get_component_status(self, score: float) -> str:
        """Get status string for a component score."""
        if score >= self._thresholds["healthy"]:
            return "healthy"
        elif score >= self._thresholds["warning"]:
            return "warning"
        elif score >= self._thresholds["degraded"]:
            return "degraded"
        else:
            return "critical"
    
    # ========== Score Analysis ==========
    
    def get_current_score(self) -> Optional[ReliabilityScore]:
        """Get the current reliability score."""
        return self._current_score
    
    def get_components(self) -> Dict[str, HealthComponent]:
        """Get current health components."""
        return self._components.copy()
    
    def get_score_trend(self) -> float:
        """
        Get the trend of the reliability score.
        
        Returns:
            Trend coefficient (positive = improving, negative = degrading)
        """
        if len(self._score_history) < 2:
            return 0.0
        
        scores = [s.score for s in self._score_history]
        x = np.arange(len(scores))
        
        # Linear regression slope
        x_mean = np.mean(x)
        y_mean = np.mean(scores)
        
        numerator = np.sum((x - x_mean) * (scores - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def is_degrading(self, threshold: float = -0.5) -> bool:
        """
        Check if reliability is degrading.
        
        Args:
            threshold: Trend threshold for degradation
            
        Returns:
            True if degrading
        """
        return self.get_score_trend() < threshold
    
    def get_weakest_components(self, n: int = 2) -> List[HealthComponent]:
        """
        Get the weakest health components.
        
        Args:
            n: Number of components to return
            
        Returns:
            List of weakest components sorted by score
        """
        sorted_components = sorted(
            self._components.values(), 
            key=lambda c: c.score
        )
        return sorted_components[:n]
    
    def get_degradation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all degradation factors.
        
        Returns:
            Dictionary with degradation analysis
        """
        if not self._current_score:
            return {"status": "no_data"}
        
        return {
            "current_score": self._current_score.score,
            "status": self._current_score.status,
            "degradation_factors": self._degradation_factors,
            "weakest_components": [
                {"name": c.name, "score": c.score, "status": c.status}
                for c in self.get_weakest_components()
            ],
            "score_trend": self.get_score_trend(),
            "is_degrading": self.is_degrading(),
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on degradation factors."""
        recommendations = []
        
        factor_recommendations = {
            "low_accuracy": "Consider retraining with recent data",
            "high_error": "Review model architecture or features",
            "declining_performance": "Monitor closely and prepare for retraining",
            "confidence_miscalibration": "Recalibrate confidence scores",
            "overconfidence": "Apply confidence calibration or use ensemble",
            "underconfidence": "Review model confidence estimation",
            "prediction_instability": "Investigate input data quality",
            "confidence_instability": "Check for data distribution changes",
            "model_staleness_predictions": "Schedule model retraining",
            "model_staleness_time": "Review model for drift",
            "feature_drift": "Investigate feature distribution changes",
            "missing_data": "Improve data quality pipeline",
            "outliers": "Review outlier handling strategy",
            "correlation_shift": "Investigate feature relationship changes",
        }
        
        for factor in self._degradation_factors:
            if factor in factor_recommendations:
                recommendations.append(factor_recommendations[factor])
        
        return list(set(recommendations))  # Remove duplicates
    
    # ========== History Analysis ==========
    
    def get_score_history(
        self, last_n: Optional[int] = None
    ) -> List[ReliabilityScore]:
        """Get historical reliability scores."""
        history = list(self._score_history)
        if last_n:
            history = history[-last_n:]
        return history
    
    def get_component_history(
        self, component: str
    ) -> Optional[RollingWindow]:
        """Get history for a specific component."""
        return self._component_history.get(component)
    
    def predict_future_score(
        self, steps_ahead: int = 10
    ) -> Tuple[float, float]:
        """
        Predict future reliability score based on trend.
        
        Args:
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Tuple of (predicted_score, confidence_in_prediction)
        """
        if len(self._score_history) < 5:
            current = self._current_score.score if self._current_score else 100.0
            return current, 0.3
        
        scores = np.array([s.score for s in self._score_history])
        trend = self.get_score_trend()
        
        # Simple linear extrapolation
        current = scores[-1]
        predicted = current + trend * steps_ahead
        predicted = max(0, min(100, predicted))
        
        # Confidence based on trend consistency
        if len(scores) > 10:
            recent_half = scores[-len(scores)//2:]
            older_half = scores[:len(scores)//2]
            
            recent_trend = np.mean(np.diff(recent_half))
            older_trend = np.mean(np.diff(older_half))
            
            # If trends are consistent, higher confidence
            trend_consistency = 1 - min(1, abs(recent_trend - older_trend) / (abs(trend) + 0.1))
            confidence = 0.3 + 0.5 * trend_consistency
        else:
            confidence = 0.4
        
        return predicted, confidence
    
    def get_alert_status(self) -> Dict[str, Any]:
        """
        Get current alert status based on reliability score.
        
        Returns:
            Dictionary with alert information
        """
        if not self._current_score:
            return {"level": "unknown", "message": "No score computed yet"}
        
        score = self._current_score.score
        status = self._current_score.status
        
        alert_levels = {
            "healthy": {
                "level": "info",
                "severity": Severity.LOW,
                "message": "Model is operating normally",
            },
            "warning": {
                "level": "warning",
                "severity": Severity.MEDIUM,
                "message": "Model showing early signs of degradation",
            },
            "degraded": {
                "level": "alert",
                "severity": Severity.HIGH,
                "message": "Model performance significantly degraded",
            },
            "critical": {
                "level": "critical",
                "severity": Severity.CRITICAL,
                "message": "Model requires immediate attention",
            },
        }
        
        alert = alert_levels.get(status, alert_levels["critical"])
        
        return {
            "level": alert["level"],
            "severity": alert["severity"].name,
            "message": alert["message"],
            "score": score,
            "status": status,
            "degradation_factors": self._degradation_factors,
            "timestamp": datetime.now().isoformat(),
        }
    
    def reset(self) -> None:
        """Reset all score history and state."""
        self._score_history.clear()
        for window in self._component_history.values():
            window.clear()
        self._current_score = None
        self._components = {}
        self._degradation_factors = []
