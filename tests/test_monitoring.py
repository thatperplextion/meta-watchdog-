"""
Unit tests for monitoring module.
"""

import pytest
import numpy as np
from datetime import datetime

from meta_watchdog.monitoring.metrics import (
    MetricType,
    RollingWindow,
    MetricAggregator,
    compute_accuracy,
    compute_calibration_error,
)
from meta_watchdog.monitoring.performance_monitor import PerformanceMonitor
from meta_watchdog.monitoring.reliability_scorer import ReliabilityScoringEngine


class TestMetricType:
    """Tests for MetricType enum."""
    
    def test_values(self):
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.CONFIDENCE.value == "confidence"
        assert MetricType.CALIBRATION_ERROR.value == "calibration_error"


class TestRollingWindow:
    """Tests for RollingWindow."""
    
    def test_creation(self):
        window = RollingWindow(window_size=10)
        assert window.window_size == 10
        assert len(window.values) == 0
    
    def test_add_values(self):
        window = RollingWindow(window_size=5)
        
        for i in range(3):
            window.add(float(i))
        
        assert len(window.values) == 3
        assert window.mean() == 1.0  # (0 + 1 + 2) / 3
    
    def test_window_overflow(self):
        window = RollingWindow(window_size=3)
        
        for i in range(5):
            window.add(float(i))
        
        # Should only keep last 3 values: 2, 3, 4
        assert len(window.values) == 3
        assert window.mean() == 3.0
    
    def test_statistics(self):
        window = RollingWindow(window_size=10)
        
        for v in [1, 2, 3, 4, 5]:
            window.add(float(v))
        
        assert window.mean() == 3.0
        assert window.std() == pytest.approx(np.std([1, 2, 3, 4, 5]), rel=1e-5)
        assert window.min() == 1.0
        assert window.max() == 5.0
    
    def test_trend(self):
        window = RollingWindow(window_size=10)
        
        # Add increasing values
        for v in [1, 2, 3, 4, 5]:
            window.add(float(v))
        
        trend = window.trend()
        assert trend > 0  # Positive trend
    
    def test_empty_window(self):
        window = RollingWindow(window_size=10)
        
        assert window.mean() == 0.0
        assert window.std() == 0.0


class TestMetricAggregator:
    """Tests for MetricAggregator."""
    
    def test_add_metric(self):
        aggregator = MetricAggregator()
        
        aggregator.add_metric(MetricType.ACCURACY, 0.85)
        aggregator.add_metric(MetricType.ACCURACY, 0.90)
        
        assert MetricType.ACCURACY in aggregator._windows
    
    def test_get_statistics(self):
        aggregator = MetricAggregator()
        
        for acc in [0.80, 0.85, 0.90]:
            aggregator.add_metric(MetricType.ACCURACY, acc)
        
        stats = aggregator.get_statistics(MetricType.ACCURACY)
        
        assert stats["mean"] == pytest.approx(0.85, rel=1e-5)
        assert stats["count"] == 3
    
    def test_summary(self):
        aggregator = MetricAggregator()
        
        aggregator.add_metric(MetricType.ACCURACY, 0.85)
        aggregator.add_metric(MetricType.CONFIDENCE, 0.75)
        
        summary = aggregator.get_summary()
        
        assert "accuracy" in summary
        assert "confidence" in summary


class TestMetricFunctions:
    """Tests for metric computation functions."""
    
    def test_compute_accuracy(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])  # 4/5 correct
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 0.8
    
    def test_compute_calibration_error(self):
        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        is_correct = np.array([True, True, False, True])
        
        error = compute_calibration_error(confidences, is_correct)
        
        # Should be some positive value
        assert 0 <= error <= 1


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""
    
    def test_creation(self):
        monitor = PerformanceMonitor()
        assert monitor._observation_count == 0
    
    def test_observe(self):
        monitor = PerformanceMonitor()
        
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1])  # 3/4 correct
        conf = np.array([0.8, 0.9, 0.7, 0.6])
        
        monitor.observe(y_true, y_pred, conf)
        
        assert monitor._observation_count == 1
    
    def test_get_metrics(self):
        monitor = PerformanceMonitor()
        
        # Add some observations
        for _ in range(5):
            y_true = np.array([0, 1, 1, 0])
            y_pred = np.array([0, 1, 1, 0])
            conf = np.array([0.9, 0.85, 0.8, 0.9])
            monitor.observe(y_true, y_pred, conf)
        
        metrics = monitor.get_metrics()
        
        assert metrics.accuracy == 1.0  # All correct
    
    def test_calibration_status(self):
        monitor = PerformanceMonitor()
        
        # Add observations with calibration data
        for _ in range(5):
            y_true = np.array([0, 1, 1, 0])
            y_pred = np.array([0, 1, 1, 0])
            conf = np.array([0.9, 0.9, 0.9, 0.9])
            monitor.observe(y_true, y_pred, conf)
        
        status = monitor.get_calibration_status()
        
        assert "is_well_calibrated" in status
        assert "avg_confidence" in status
    
    def test_reset(self):
        monitor = PerformanceMonitor()
        
        monitor.observe(
            np.array([0, 1]), 
            np.array([0, 1]), 
            np.array([0.9, 0.9])
        )
        
        monitor.reset()
        
        assert monitor._observation_count == 0


class TestReliabilityScoringEngine:
    """Tests for ReliabilityScoringEngine."""
    
    def test_creation(self):
        scorer = ReliabilityScoringEngine()
        assert scorer is not None
    
    def test_compute_score_minimal(self):
        scorer = ReliabilityScoringEngine()
        
        from meta_watchdog.core.data_structures import PerformanceMetrics
        metrics = PerformanceMetrics(accuracy=0.85)
        
        score = scorer.compute_score(metrics)
        
        assert score.score >= 0
        assert score.score <= 100
    
    def test_compute_score_full(self):
        scorer = ReliabilityScoringEngine()
        
        from meta_watchdog.core.data_structures import PerformanceMetrics
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90
        )
        
        # Simulate some history
        for _ in range(10):
            scorer.compute_score(metrics)
        
        score = scorer.compute_score(metrics)
        
        assert score.score > 0
        assert score.performance_score > 0
    
    def test_degradation_tracking(self):
        scorer = ReliabilityScoringEngine()
        
        from meta_watchdog.core.data_structures import PerformanceMetrics
        
        # Good metrics initially
        for _ in range(5):
            metrics = PerformanceMetrics(accuracy=0.90)
            scorer.compute_score(metrics)
        
        # Degraded metrics
        for _ in range(5):
            metrics = PerformanceMetrics(accuracy=0.60)
            scorer.compute_score(metrics)
        
        degradation = scorer.get_degradation_rate()
        
        # Should detect degradation
        assert degradation is not None
    
    def test_get_health_report(self):
        scorer = ReliabilityScoringEngine()
        
        from meta_watchdog.core.data_structures import PerformanceMetrics
        metrics = PerformanceMetrics(accuracy=0.85)
        
        scorer.compute_score(metrics)
        
        report = scorer.get_health_report()
        
        assert "current_score" in report
        assert "health_status" in report
