"""
Base Prediction Model

Abstract base class for ML models that can be monitored by Meta-Watchdog.
This class provides the foundation for any prediction model to become self-aware.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import hashlib
import json
from datetime import datetime

from meta_watchdog.core.data_structures import Prediction, PredictionBatch


class ModelType(Enum):
    """Type of ML model."""
    CLASSIFICATION = auto()
    REGRESSION = auto()
    MULTI_CLASS = auto()
    MULTI_LABEL = auto()
    RANKING = auto()


class BaseModel(ABC):
    """
    Abstract base class for all prediction models in Meta-Watchdog.
    
    This class defines the interface that any ML model must implement
    to participate in the self-aware monitoring system.
    
    Key Features:
    - Dataset-agnostic design
    - Mandatory confidence scoring
    - Built-in prediction tracking
    - Feature importance awareness
    
    Subclasses must implement:
    - _predict_impl: Core prediction logic
    - _get_confidence_impl: Confidence estimation logic
    """
    
    def __init__(
        self,
        model_type: ModelType = ModelType.REGRESSION,
        confidence_method: str = "default",
        track_predictions: bool = True,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the base model.
        
        Args:
            model_type: Type of prediction task
            confidence_method: Method for computing confidence
                - "default": Use model's native confidence
                - "ensemble": Use ensemble-based uncertainty
                - "calibrated": Use calibrated probabilities
            track_predictions: Whether to track predictions internally
            model_name: Optional name for the model
        """
        self._model_type = model_type
        self._confidence_method = confidence_method
        self._track_predictions = track_predictions
        self._model_name = model_name or self.__class__.__name__
        
        # Internal state
        self._is_fitted = False
        self._prediction_count = 0
        self._prediction_history: List[Prediction] = []
        self._feature_names: Optional[List[str]] = None
        self._feature_importance: Optional[Dict[str, float]] = None
        
        # Model metadata
        self._created_at = datetime.now()
        self._last_trained_at: Optional[datetime] = None
        self._training_samples: int = 0
        self._version = "1.0.0"
        
        # Confidence calibration state
        self._confidence_bias: float = 0.0
        self._confidence_scale: float = 1.0
    
    # ========== Abstract Methods (Must Implement) ==========
    
    @abstractmethod
    def _predict_impl(
        self, X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Internal prediction implementation.
        
        Subclasses must override this with actual prediction logic.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Raw predictions (n_samples,) or (n_samples, n_outputs)
        """
        pass
    
    @abstractmethod
    def _get_confidence_impl(
        self, X: NDArray[np.floating], predictions: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Internal confidence computation.
        
        Subclasses must override this to provide confidence scores.
        
        Args:
            X: Input features
            predictions: Predictions made for X
            
        Returns:
            Confidence scores (n_samples,) in range [0, 1]
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        **kwargs
    ) -> "BaseModel":
        """
        Train the model on data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    # ========== Public Interface ==========
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Make predictions on input data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,) or (n_samples, n_outputs)
        """
        self._validate_fitted()
        X = self._validate_input(X)
        
        predictions = self._predict_impl(X)
        self._prediction_count += len(X)
        
        return predictions
    
    def predict_with_confidence(
        self, X: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Make predictions with confidence scores.
        
        This is the KEY method for self-awareness. The model expresses
        its uncertainty about each prediction.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Tuple of:
                - predictions: (n_samples,) or (n_samples, n_outputs)
                - confidence: (n_samples,) values in [0, 1]
        """
        self._validate_fitted()
        X = self._validate_input(X)
        
        # Get predictions
        predictions = self._predict_impl(X)
        
        # Get raw confidence
        raw_confidence = self._get_confidence_impl(X, predictions)
        
        # Apply calibration if available
        confidence = self._calibrate_confidence(raw_confidence)
        
        # Track predictions if enabled
        if self._track_predictions:
            self._record_predictions(X, predictions, confidence)
        
        self._prediction_count += len(X)
        
        return predictions, confidence
    
    def predict_batch(
        self, X: NDArray[np.floating], batch_id: Optional[str] = None
    ) -> PredictionBatch:
        """
        Make predictions and return as a PredictionBatch.
        
        Args:
            X: Input features
            batch_id: Optional batch identifier
            
        Returns:
            PredictionBatch with predictions and metadata
        """
        predictions, confidence = self.predict_with_confidence(X)
        
        batch_id = batch_id or self._generate_batch_id()
        
        prediction_objects = [
            Prediction(
                value=float(pred) if np.isscalar(pred) else pred,
                confidence=float(conf),
                input_hash=self._hash_input(X[i]),
                prediction_id=f"{batch_id}_{i}"
            )
            for i, (pred, conf) in enumerate(zip(predictions, confidence))
        ]
        
        return PredictionBatch(
            predictions=prediction_objects,
            batch_id=batch_id
        )
    
    # ========== Confidence Calibration ==========
    
    def calibrate_confidence(
        self,
        X_val: NDArray[np.floating],
        y_val: NDArray[np.floating]
    ) -> None:
        """
        Calibrate confidence scores using validation data.
        
        This adjusts confidence to better reflect actual accuracy.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        predictions, confidence = self.predict_with_confidence(X_val)
        
        # Compute actual accuracy for confidence bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        actual_accuracies = []
        mean_confidences = []
        
        for i in range(n_bins):
            mask = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                if self._model_type == ModelType.CLASSIFICATION:
                    actual_acc = np.mean(predictions[mask] == y_val[mask])
                else:
                    # For regression, use a tolerance-based accuracy
                    tolerance = np.std(y_val) * 0.1
                    actual_acc = np.mean(np.abs(predictions[mask] - y_val[mask]) < tolerance)
                
                actual_accuracies.append(actual_acc)
                mean_confidences.append(np.mean(confidence[mask]))
        
        if len(actual_accuracies) >= 2:
            # Simple linear calibration
            actual_accuracies = np.array(actual_accuracies)
            mean_confidences = np.array(mean_confidences)
            
            self._confidence_scale = np.std(actual_accuracies) / (np.std(mean_confidences) + 1e-8)
            self._confidence_bias = np.mean(actual_accuracies) - self._confidence_scale * np.mean(mean_confidences)
    
    def _calibrate_confidence(
        self, confidence: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply calibration to confidence scores."""
        calibrated = self._confidence_scale * confidence + self._confidence_bias
        return np.clip(calibrated, 0.0, 1.0)
    
    # ========== Feature Importance ==========
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available.
        """
        return self._feature_importance
    
    def set_feature_names(self, names: List[str]) -> None:
        """
        Set feature names for better interpretability.
        
        Args:
            names: List of feature names
        """
        self._feature_names = names
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        """Get feature names."""
        return self._feature_names
    
    # ========== Model Properties ==========
    
    @property
    def model_type(self) -> str:
        """Return the type of model as string."""
        return self._model_type.name.lower()
    
    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_fitted
    
    @property
    def prediction_count(self) -> int:
        """Get total number of predictions made."""
        return self._prediction_count
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
    
    @property
    def version(self) -> str:
        """Get model version."""
        return self._version
    
    @property
    def age_in_predictions(self) -> int:
        """Get number of predictions since last training."""
        return self._prediction_count
    
    @property
    def training_samples(self) -> int:
        """Get number of samples used in training."""
        return self._training_samples
    
    # ========== Prediction History ==========
    
    def get_prediction_history(
        self, last_n: Optional[int] = None
    ) -> List[Prediction]:
        """
        Get prediction history.
        
        Args:
            last_n: Return only last N predictions
            
        Returns:
            List of Prediction objects
        """
        if last_n is None:
            return self._prediction_history.copy()
        return self._prediction_history[-last_n:]
    
    def update_with_ground_truth(
        self, prediction_id: str, actual_value: Union[float, int, str]
    ) -> bool:
        """
        Update a prediction with ground truth.
        
        Args:
            prediction_id: ID of the prediction to update
            actual_value: The actual value
            
        Returns:
            True if prediction was found and updated
        """
        for pred in self._prediction_history:
            if pred.prediction_id == prediction_id:
                pred.actual_value = actual_value
                
                # Compute correctness/error
                if self._model_type == ModelType.CLASSIFICATION:
                    pred.was_correct = (pred.value == actual_value)
                    pred.error = 0.0 if pred.was_correct else 1.0
                else:
                    pred.error = abs(float(pred.value) - float(actual_value))
                    pred.was_correct = None  # Not applicable for regression
                
                return True
        return False
    
    def clear_history(self) -> None:
        """Clear prediction history."""
        self._prediction_history = []
    
    # ========== Serialization ==========
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_name": self._model_name,
            "model_type": self.model_type,
            "version": self._version,
            "is_fitted": self._is_fitted,
            "prediction_count": self._prediction_count,
            "training_samples": self._training_samples,
            "created_at": self._created_at.isoformat(),
            "last_trained_at": self._last_trained_at.isoformat() if self._last_trained_at else None,
            "feature_names": self._feature_names,
            "confidence_method": self._confidence_method,
        }
    
    # ========== Internal Helpers ==========
    
    def _validate_fitted(self) -> None:
        """Ensure model is fitted before prediction."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
    
    def _validate_input(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Validate and preprocess input."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X.astype(np.float64)
    
    def _record_predictions(
        self,
        X: NDArray[np.floating],
        predictions: NDArray[np.floating],
        confidence: NDArray[np.floating]
    ) -> None:
        """Record predictions in history."""
        batch_id = self._generate_batch_id()
        
        for i, (pred, conf) in enumerate(zip(predictions, confidence)):
            prediction = Prediction(
                value=float(pred) if np.isscalar(pred) or pred.ndim == 0 else pred.tolist(),
                confidence=float(conf),
                input_hash=self._hash_input(X[i]),
                prediction_id=f"{batch_id}_{i}"
            )
            self._prediction_history.append(prediction)
        
        # Limit history size to prevent memory issues
        max_history = 10000
        if len(self._prediction_history) > max_history:
            self._prediction_history = self._prediction_history[-max_history:]
    
    def _hash_input(self, x: NDArray[np.floating]) -> str:
        """Create a hash of input for tracking."""
        return hashlib.md5(x.tobytes()).hexdigest()[:12]
    
    def _generate_batch_id(self) -> str:
        """Generate a unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"batch_{timestamp}"
    
    def _mark_fitted(self, n_samples: int) -> None:
        """Mark model as fitted after training."""
        self._is_fitted = True
        self._last_trained_at = datetime.now()
        self._training_samples = n_samples
        self._prediction_count = 0  # Reset prediction count after retraining


class ConfidenceEstimator:
    """
    Utility class for estimating prediction confidence.
    
    Provides various methods for computing confidence when the
    underlying model doesn't provide native uncertainty estimates.
    """
    
    @staticmethod
    def from_probabilities(probabilities: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute confidence from class probabilities.
        
        For classification: max probability
        
        Args:
            probabilities: Class probabilities (n_samples, n_classes)
            
        Returns:
            Confidence scores (n_samples,)
        """
        if probabilities.ndim == 1:
            return probabilities
        return np.max(probabilities, axis=1)
    
    @staticmethod
    def from_ensemble_variance(
        ensemble_predictions: List[NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """
        Compute confidence from ensemble prediction variance.
        
        Lower variance = higher confidence.
        
        Args:
            ensemble_predictions: List of predictions from ensemble members
            
        Returns:
            Confidence scores (n_samples,)
        """
        stacked = np.stack(ensemble_predictions, axis=0)
        variance = np.var(stacked, axis=0)
        
        # Normalize variance to [0, 1] and invert (low variance = high confidence)
        max_var = np.max(variance) + 1e-8
        confidence = 1.0 - (variance / max_var)
        
        return confidence
    
    @staticmethod
    def from_distance_to_training(
        X: NDArray[np.floating],
        X_train: NDArray[np.floating],
        k: int = 5
    ) -> NDArray[np.floating]:
        """
        Compute confidence based on distance to training data.
        
        Closer to training data = higher confidence.
        
        Args:
            X: Query points
            X_train: Training data
            k: Number of nearest neighbors
            
        Returns:
            Confidence scores (n_samples,)
        """
        from scipy.spatial.distance import cdist
        
        distances = cdist(X, X_train)
        k_nearest = np.sort(distances, axis=1)[:, :k]
        mean_distances = np.mean(k_nearest, axis=1)
        
        # Normalize and invert
        max_dist = np.max(mean_distances) + 1e-8
        confidence = 1.0 - (mean_distances / max_dist)
        
        return confidence
    
    @staticmethod
    def from_prediction_stability(
        X: NDArray[np.floating],
        predict_fn,
        n_perturbations: int = 10,
        noise_scale: float = 0.01
    ) -> NDArray[np.floating]:
        """
        Compute confidence based on prediction stability under perturbation.
        
        Stable predictions = higher confidence.
        
        Args:
            X: Input data
            predict_fn: Function to make predictions
            n_perturbations: Number of perturbations
            noise_scale: Scale of Gaussian noise
            
        Returns:
            Confidence scores (n_samples,)
        """
        predictions = []
        
        for _ in range(n_perturbations):
            noise = np.random.normal(0, noise_scale, X.shape)
            perturbed = X + noise
            pred = predict_fn(perturbed)
            predictions.append(pred)
        
        stacked = np.stack(predictions, axis=0)
        variance = np.var(stacked, axis=0)
        
        if variance.ndim > 1:
            variance = np.mean(variance, axis=1)
        
        max_var = np.max(variance) + 1e-8
        confidence = 1.0 - (variance / max_var)
        
        return confidence
