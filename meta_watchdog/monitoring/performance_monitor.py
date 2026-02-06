"""
Performance Monitoring Module

Continuously monitors model performance, tracking metrics over time
and detecting patterns that indicate potential issues.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
import numpy as np
from numpy.typing import NDArray

from meta_watchdog.core.interfaces import Monitor
from meta_watchdog.core.data_structures import (
    Prediction,
    PredictionBatch,
    PerformanceMetrics,
)
from meta_watchdog.monitoring.metrics import (
    MetricType,
    MetricAggregator,
    RollingWindow,
    compute_accuracy,
    compute_mae,
    compute_rmse,
    compute_confidence_accuracy_gap,
    compute_calibration_error,
    compute_brier_score,
)


@dataclass
class ConfidenceCorrectnessPair:
    """Tracks confidence vs actual correctness for calibration analysis."""
    confidence: float
    was_correct: bool
    error: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor(Monitor):
    """
    Monitors model performance and health over time.
    
    Key Responsibilities:
    - Track predictions and outcomes
    - Compute rolling performance metrics
    - Detect confidence-correctness mismatches
    - Identify performance degradation trends
    
    This is the "eyes" of the self-aware system.
    """
    
    def __init__(
        self,
        model_type: str = "classification",
        window_size: int = 100,
        confidence_threshold: float = 0.7,
        performance_threshold: float = 0.8,
    ):
        """
        Initialize performance monitor.
        
        Args:
            model_type: "classification" or "regression"
            window_size: Size of rolling window for metrics
            confidence_threshold: Threshold for "high confidence"
            performance_threshold: Threshold for acceptable performance
        """
        self.model_type = model_type
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.performance_threshold = performance_threshold
        
        # Core tracking
        self._predictions: deque = deque(maxlen=window_size * 10)
        self._confidence_correctness: deque = deque(maxlen=window_size * 10)
        
        # Metric aggregator
        self._metrics = MetricAggregator(window_size=window_size)
        
        # Initialize default metrics
        self._initialize_metrics()
        
        # State
        self._total_predictions = 0
        self._predictions_with_ground_truth = 0
        self._last_update = datetime.now()
    
    def _initialize_metrics(self) -> None:
        """Initialize all tracked metrics."""
        metrics_to_track = [
            "accuracy", "mae", "rmse",
            "mean_confidence", "confidence_std",
            "confidence_accuracy_gap", "calibration_error",
            "overconfidence_ratio", "underconfidence_ratio",
            "prediction_variance", "confidence_variance"
        ]
        for metric in metrics_to_track:
            self._metrics.register_metric(metric)
    
    # ========== Monitor Interface Implementation ==========
    
    def observe(self, data: Dict[str, Any]) -> None:
        """
        Record an observation from the system.
        
        Expected data format:
        {
            "prediction": Prediction or value,
            "confidence": float,
            "actual": optional ground truth,
            "features": optional input features,
            "timestamp": optional datetime
        }
        """
        timestamp = data.get("timestamp", datetime.now())
        
        # Handle Prediction objects
        if isinstance(data.get("prediction"), Prediction):
            pred = data["prediction"]
            prediction_value = pred.value
            confidence = pred.confidence
            actual = pred.actual_value
        else:
            prediction_value = data.get("prediction")
            confidence = data.get("confidence", 0.5)
            actual = data.get("actual")
        
        # Store prediction
        pred_record = {
            "prediction": prediction_value,
            "confidence": confidence,
            "actual": actual,
            "timestamp": timestamp,
            "features": data.get("features"),
        }
        self._predictions.append(pred_record)
        self._total_predictions += 1
        
        # If we have ground truth, compute correctness
        if actual is not None:
            self._predictions_with_ground_truth += 1
            
            if self.model_type == "classification":
                was_correct = (prediction_value == actual)
                error = 0.0 if was_correct else 1.0
            else:
                was_correct = None
                error = abs(float(prediction_value) - float(actual))
            
            self._confidence_correctness.append(
                ConfidenceCorrectnessPair(
                    confidence=confidence,
                    was_correct=was_correct,
                    error=error,
                    timestamp=timestamp
                )
            )
            
            # Update metrics
            self._update_metrics()
        
        self._last_update = timestamp
    
    def observe_batch(
        self,
        predictions: NDArray,
        confidences: NDArray,
        actuals: Optional[NDArray] = None,
        features: Optional[NDArray] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a batch of observations.
        
        Args:
            predictions: Array of predictions
            confidences: Array of confidence scores
            actuals: Optional array of ground truth values
            features: Optional array of input features
            timestamp: Optional timestamp for all observations
        """
        ts = timestamp or datetime.now()
        
        for i in range(len(predictions)):
            data = {
                "prediction": predictions[i],
                "confidence": confidences[i],
                "timestamp": ts,
            }
            
            if actuals is not None:
                data["actual"] = actuals[i]
            
            if features is not None:
                data["features"] = features[i]
            
            self.observe(data)
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current monitoring state.
        
        Returns comprehensive snapshot of current performance.
        """
        latest_metrics = self.compute_current_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": self._total_predictions,
            "predictions_with_ground_truth": self._predictions_with_ground_truth,
            "window_size": self.window_size,
            "model_type": self.model_type,
            "metrics": latest_metrics,
            "trends": self._metrics.get_trends(),
            "health_indicators": self._metrics.compute_health_indicators(),
            "is_degrading": self.is_performance_degrading(),
            "calibration_status": self.get_calibration_status(),
        }
    
    def get_history(
        self, window_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get historical observations."""
        predictions = list(self._predictions)
        if window_size:
            predictions = predictions[-window_size:]
        return predictions
    
    def reset(self) -> None:
        """Clear all stored observations and reset state."""
        self._predictions.clear()
        self._confidence_correctness.clear()
        self._metrics.reset()
        self._initialize_metrics()
        self._total_predictions = 0
        self._predictions_with_ground_truth = 0
    
    # ========== Metric Computation ==========
    
    def _update_metrics(self) -> None:
        """Update all tracked metrics based on recent observations."""
        if len(self._confidence_correctness) < 2:
            return
        
        recent = list(self._confidence_correctness)[-self.window_size:]
        
        confidences = np.array([cc.confidence for cc in recent])
        errors = np.array([cc.error for cc in recent if cc.error is not None])
        
        # Record confidence metrics
        self._metrics.record("mean_confidence", float(np.mean(confidences)))
        self._metrics.record("confidence_std", float(np.std(confidences)))
        self._metrics.record("confidence_variance", float(np.var(confidences)))
        
        if self.model_type == "classification":
            was_correct = np.array([cc.was_correct for cc in recent])
            
            # Accuracy
            accuracy = float(np.mean(was_correct))
            self._metrics.record("accuracy", accuracy)
            
            # Confidence-accuracy gap
            gap = compute_confidence_accuracy_gap(confidences, was_correct)
            self._metrics.record("confidence_accuracy_gap", gap)
            
            # Calibration error
            if len(was_correct) >= 10:
                cal_error = compute_calibration_error(confidences, was_correct)
                self._metrics.record("calibration_error", cal_error)
            
            # Overconfidence ratio: wrong predictions with high confidence
            wrong_high_conf = np.sum(
                (~was_correct) & (confidences >= self.confidence_threshold)
            )
            overconf_ratio = wrong_high_conf / max(1, np.sum(~was_correct))
            self._metrics.record("overconfidence_ratio", float(overconf_ratio))
            
            # Underconfidence ratio: correct predictions with low confidence
            correct_low_conf = np.sum(
                was_correct & (confidences < self.confidence_threshold)
            )
            underconf_ratio = correct_low_conf / max(1, np.sum(was_correct))
            self._metrics.record("underconfidence_ratio", float(underconf_ratio))
        
        else:  # Regression
            if len(errors) > 0:
                self._metrics.record("mae", float(np.mean(errors)))
                self._metrics.record("rmse", float(np.sqrt(np.mean(errors ** 2))))
        
        # Prediction variance (if we can compute it)
        predictions = [p["prediction"] for p in list(self._predictions)[-self.window_size:]]
        try:
            pred_array = np.array(predictions, dtype=float)
            self._metrics.record("prediction_variance", float(np.var(pred_array)))
        except (TypeError, ValueError):
            pass
    
    def compute_current_metrics(self) -> PerformanceMetrics:
        """
        Compute current performance metrics.
        
        Returns:
            PerformanceMetrics object with all current values
        """
        stats = self._metrics.get_all_statistics()
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            window_size=min(len(self._confidence_correctness), self.window_size),
        )
        
        # Fill in available metrics
        if "accuracy" in stats:
            metrics.accuracy = stats["accuracy"].get("mean")
        if "mae" in stats:
            metrics.mean_error = stats["mae"].get("mean")
        if "rmse" in stats:
            metrics.rmse = stats["rmse"].get("mean")
        
        if "mean_confidence" in stats:
            metrics.mean_confidence = stats["mean_confidence"].get("mean", 0.0)
        if "confidence_std" in stats:
            metrics.confidence_std = stats["confidence_std"].get("mean", 0.0)
        
        if "confidence_accuracy_gap" in stats:
            metrics.confidence_accuracy_gap = stats["confidence_accuracy_gap"].get("mean", 0.0)
        if "overconfidence_ratio" in stats:
            metrics.overconfident_ratio = stats["overconfidence_ratio"].get("mean", 0.0)
        if "underconfidence_ratio" in stats:
            metrics.underconfident_ratio = stats["underconfidence_ratio"].get("mean", 0.0)
        
        if "prediction_variance" in stats:
            metrics.prediction_variance = stats["prediction_variance"].get("mean", 0.0)
        if "confidence_variance" in stats:
            metrics.confidence_variance = stats["confidence_variance"].get("mean", 0.0)
        
        # Trends
        trends = self._metrics.get_trends()
        metrics.accuracy_trend = trends.get("accuracy", 0.0)
        metrics.confidence_trend = trends.get("mean_confidence", 0.0)
        
        return metrics
    
    # ========== Performance Analysis ==========
    
    def is_performance_degrading(self) -> bool:
        """
        Check if performance is showing degradation.
        
        Returns:
            True if degradation detected
        """
        degrading_metrics = self._metrics.get_degrading_metrics()
        
        # Performance is degrading if key metrics are degrading
        key_metrics = ["accuracy", "mae", "rmse"]
        return any(m in degrading_metrics for m in key_metrics)
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get calibration status.
        
        Returns:
            Dictionary with calibration analysis
        """
        stats = self._metrics.get_all_statistics()
        
        cal_error = stats.get("calibration_error", {}).get("mean", 0.0)
        overconf = stats.get("overconfidence_ratio", {}).get("mean", 0.0)
        underconf = stats.get("underconfidence_ratio", {}).get("mean", 0.0)
        
        # Determine status
        if cal_error < 0.05 and overconf < 0.1 and underconf < 0.1:
            status = "well_calibrated"
        elif cal_error > 0.2 or overconf > 0.3:
            status = "poorly_calibrated"
        else:
            status = "moderately_calibrated"
        
        return {
            "status": status,
            "calibration_error": cal_error,
            "overconfidence_ratio": overconf,
            "underconfidence_ratio": underconf,
            "recommendation": self._get_calibration_recommendation(status),
        }
    
    def _get_calibration_recommendation(self, status: str) -> str:
        """Get recommendation based on calibration status."""
        if status == "well_calibrated":
            return "Model confidence is well-calibrated. No action needed."
        elif status == "poorly_calibrated":
            return "Model confidence needs recalibration. Consider using calibration techniques."
        else:
            return "Model confidence is moderately calibrated. Monitor for further drift."
    
    def get_confidence_correctness_analysis(self) -> Dict[str, Any]:
        """
        Analyze relationship between confidence and correctness.
        
        Returns:
            Detailed analysis of confidence-correctness relationship
        """
        if len(self._confidence_correctness) < 10:
            return {"status": "insufficient_data", "message": "Need more predictions with ground truth"}
        
        recent = list(self._confidence_correctness)[-self.window_size:]
        
        confidences = np.array([cc.confidence for cc in recent])
        
        analysis = {
            "sample_size": len(recent),
            "confidence_distribution": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
            },
        }
        
        if self.model_type == "classification":
            was_correct = np.array([cc.was_correct for cc in recent])
            
            # Bin analysis
            bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
            bin_analysis = []
            
            for low, high in bins:
                mask = (confidences >= low) & (confidences < high)
                if np.sum(mask) > 0:
                    bin_analysis.append({
                        "confidence_range": f"{low:.1f}-{high:.1f}",
                        "count": int(np.sum(mask)),
                        "accuracy": float(np.mean(was_correct[mask])),
                        "expected_accuracy": (low + high) / 2,
                    })
            
            analysis["bin_analysis"] = bin_analysis
            analysis["overall_accuracy"] = float(np.mean(was_correct))
        
        else:  # Regression
            errors = np.array([cc.error for cc in recent if cc.error is not None])
            
            if len(errors) > 0:
                # Check if low confidence correlates with high error
                analysis["error_distribution"] = {
                    "mean": float(np.mean(errors)),
                    "std": float(np.std(errors)),
                    "p90": float(np.percentile(errors, 90)),
                    "max": float(np.max(errors)),
                }
                
                # Correlation between confidence and error
                correlation = np.corrcoef(confidences[:len(errors)], errors)[0, 1]
                analysis["confidence_error_correlation"] = float(correlation)
        
        return analysis
    
    def get_metric_trends(self) -> Dict[str, Dict[str, float]]:
        """
        Get trends for all tracked metrics.
        
        Returns:
            Dictionary with trend information for each metric
        """
        result = {}
        
        for name in ["accuracy", "mae", "rmse", "mean_confidence", 
                     "confidence_accuracy_gap", "calibration_error"]:
            window = self._metrics.get_window(name)
            if window and len(window) > 1:
                result[name] = {
                    "trend": window.trend(),
                    "trend_strength": window.trend_strength(),
                    "is_degrading": window.is_degrading(),
                    "recent_mean": window.mean(),
                }
        
        return result
    
    def get_anomalies(self, z_threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalous predictions.
        
        Args:
            z_threshold: Z-score threshold for anomaly
            
        Returns:
            List of anomalous predictions with details
        """
        if len(self._confidence_correctness) < 10:
            return []
        
        recent = list(self._confidence_correctness)[-self.window_size:]
        confidences = np.array([cc.confidence for cc in recent])
        
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        if std_conf == 0:
            return []
        
        anomalies = []
        
        for i, cc in enumerate(recent):
            z_score = (cc.confidence - mean_conf) / std_conf
            
            is_anomaly = False
            reason = ""
            
            # Unusually low confidence
            if z_score < -z_threshold:
                is_anomaly = True
                reason = "unusually_low_confidence"
            
            # Unusually high confidence with wrong prediction
            if self.model_type == "classification":
                if z_score > z_threshold and not cc.was_correct:
                    is_anomaly = True
                    reason = "overconfident_wrong_prediction"
            
            if is_anomaly:
                anomalies.append({
                    "index": i,
                    "confidence": cc.confidence,
                    "z_score": float(z_score),
                    "was_correct": cc.was_correct,
                    "reason": reason,
                    "timestamp": cc.timestamp.isoformat(),
                })
        
        return anomalies
    
    # ========== Summary Methods ==========
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of monitoring state.
        
        Returns:
            Dictionary with full monitoring summary
        """
        return {
            "overview": {
                "total_predictions": self._total_predictions,
                "predictions_with_feedback": self._predictions_with_ground_truth,
                "feedback_rate": (
                    self._predictions_with_ground_truth / max(1, self._total_predictions)
                ),
                "model_type": self.model_type,
                "last_update": self._last_update.isoformat(),
            },
            "current_metrics": self.compute_current_metrics().__dict__,
            "calibration": self.get_calibration_status(),
            "trends": self.get_metric_trends(),
            "health_indicators": self._metrics.compute_health_indicators(),
            "anomalies": self.get_anomalies(),
        }
