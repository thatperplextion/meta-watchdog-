"""
Unit tests for core module.
"""

import pytest
import numpy as np
from datetime import datetime

from meta_watchdog.core.data_structures import (
    Prediction,
    PredictionBatch,
    PerformanceMetrics,
    ReliabilityScore,
    FailurePrediction,
    RootCause,
    Recommendation,
    ActionType,
    ScenarioType,
)
from meta_watchdog.core.base_model import BaseModel, ConfidenceEstimator


class DummyModel(BaseModel):
    """Dummy model for testing."""
    
    def __init__(self):
        super().__init__("dummy_model", "1.0")
        self._weight = 0.5
    
    def _predict_impl(self, X):
        return (np.sum(X, axis=1) > 0).astype(int)
    
    def _predict_proba_impl(self, X):
        scores = 1 / (1 + np.exp(-np.sum(X, axis=1)))
        return np.column_stack([1 - scores, scores])


class TestPrediction:
    """Tests for Prediction data structure."""
    
    def test_creation(self):
        pred = Prediction(
            value=1,
            confidence=0.85,
            timestamp=datetime.now()
        )
        assert pred.value == 1
        assert pred.confidence == 0.85
    
    def test_features(self):
        pred = Prediction(
            value=0,
            confidence=0.7,
            features={"age": 25, "income": 50000}
        )
        assert pred.features["age"] == 25


class TestPredictionBatch:
    """Tests for PredictionBatch data structure."""
    
    def test_creation(self):
        batch = PredictionBatch(
            predictions=np.array([0, 1, 1, 0]),
            confidences=np.array([0.8, 0.9, 0.7, 0.85]),
            batch_id="test-001"
        )
        assert len(batch.predictions) == 4
        assert batch.batch_id == "test-001"
    
    def test_ground_truth(self):
        batch = PredictionBatch(
            predictions=np.array([0, 1, 1, 0]),
            confidences=np.array([0.8, 0.9, 0.7, 0.85]),
            ground_truth=np.array([0, 1, 0, 0])
        )
        assert batch.ground_truth is not None


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics data structure."""
    
    def test_creation(self):
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85
        )
        assert metrics.accuracy == 0.85
        assert metrics.f1_score == 0.85
    
    def test_optional_fields(self):
        metrics = PerformanceMetrics(
            accuracy=0.9,
            auc_roc=0.95,
            log_loss=0.3
        )
        assert metrics.auc_roc == 0.95
        assert metrics.precision is None


class TestReliabilityScore:
    """Tests for ReliabilityScore data structure."""
    
    def test_creation(self):
        score = ReliabilityScore(
            score=75.5,
            performance_score=0.8,
            calibration_score=0.7,
            stability_score=0.75,
            freshness_score=0.8,
            feature_health_score=0.7
        )
        assert score.score == 75.5
        assert score.performance_score == 0.8
    
    def test_components_list(self):
        score = ReliabilityScore(
            score=80.0,
            performance_score=0.85,
            calibration_score=0.75,
            stability_score=0.8,
            freshness_score=0.85,
            feature_health_score=0.75,
            component_scores={"custom": 0.9}
        )
        assert score.component_scores["custom"] == 0.9


class TestFailurePrediction:
    """Tests for FailurePrediction data structure."""
    
    def test_creation(self):
        pred = FailurePrediction(
            probability=0.65,
            failure_type="drift",
            confidence=0.8,
            contributing_factors=["feature_drift", "calibration_degradation"]
        )
        assert pred.probability == 0.65
        assert "feature_drift" in pred.contributing_factors
    
    def test_time_to_failure(self):
        pred = FailurePrediction(
            probability=0.8,
            failure_type="performance_degradation",
            confidence=0.7,
            estimated_time_to_failure=3600.0  # 1 hour
        )
        assert pred.estimated_time_to_failure == 3600.0


class TestBaseModel:
    """Tests for BaseModel."""
    
    def test_predict(self):
        model = DummyModel()
        X = np.array([[1, 2, 3], [-1, -2, -3], [0.5, 0.5, 0.5]])
        
        predictions = model.predict(X)
        
        assert len(predictions) == 3
        assert predictions[0] == 1  # Positive sum
        assert predictions[1] == 0  # Negative sum
    
    def test_predict_with_confidence(self):
        model = DummyModel()
        X = np.array([[1, 2, 3], [-1, -2, -3]])
        
        predictions, confidences = model.predict_with_confidence(X)
        
        assert len(predictions) == 2
        assert len(confidences) == 2
        assert all(0 <= c <= 1 for c in confidences)
    
    def test_prediction_history(self):
        model = DummyModel()
        X = np.array([[1, 2, 3]])
        
        model.predict(X)
        model.predict(X)
        
        assert model.prediction_count == 2
        assert len(model._prediction_history) == 2


class TestConfidenceEstimator:
    """Tests for ConfidenceEstimator."""
    
    def test_from_probabilities(self):
        proba = np.array([[0.3, 0.7], [0.1, 0.9], [0.5, 0.5]])
        
        confidences = ConfidenceEstimator.from_probabilities(proba)
        
        assert len(confidences) == 3
        assert confidences[0] == 0.7  # max(0.3, 0.7)
        assert confidences[1] == 0.9  # max(0.1, 0.9)
        assert confidences[2] == 0.5  # max(0.5, 0.5)
    
    def test_calibrate(self):
        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        is_correct = np.array([True, True, False, True])
        
        calibrated = ConfidenceEstimator.calibrate(confidences, is_correct)
        
        # Calibration should adjust confidences based on actual correctness
        assert len(calibrated) == 4
        assert all(0 <= c <= 1 for c in calibrated)


class TestScenarioType:
    """Tests for ScenarioType enum."""
    
    def test_values(self):
        assert ScenarioType.FEATURE_DRIFT.value == "feature_drift"
        assert ScenarioType.NOISE_INJECTION.value == "noise_injection"
        assert ScenarioType.OUTLIER_STORM.value == "outlier_storm"
    
    def test_all_types(self):
        types = list(ScenarioType)
        assert len(types) == 8  # 8 scenario types


class TestActionType:
    """Tests for ActionType enum."""
    
    def test_values(self):
        assert ActionType.RETRAIN.value == "retrain"
        assert ActionType.ALERT.value == "alert"
        assert ActionType.ROLLBACK.value == "rollback"
