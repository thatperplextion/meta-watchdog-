"""
Synthetic Data Generator

Generates synthetic data for testing and demonstrating Meta-Watchdog
without requiring real production data.

Supports:
- Normal operation data
- Drift scenarios
- Degradation patterns
- Failure simulations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class DataPattern(Enum):
    """Patterns for synthetic data generation."""
    NORMAL = "normal"                    # Stable, well-behaved
    GRADUAL_DRIFT = "gradual_drift"      # Slow drift over time
    SUDDEN_DRIFT = "sudden_drift"        # Abrupt distribution change
    INCREASING_NOISE = "increasing_noise"  # Growing noise levels
    PERIODIC = "periodic"                # Cyclical patterns
    DEGRADING = "degrading"              # Performance degradation
    MIXED = "mixed"                      # Combination of patterns


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 1000
    n_features: int = 10
    n_classes: int = 2
    noise_level: float = 0.1
    drift_rate: float = 0.01
    random_seed: Optional[int] = None


class SyntheticDataGenerator:
    """
    Generates synthetic data for testing Meta-Watchdog.
    
    Creates realistic data patterns including:
    - Feature distributions with controlled drift
    - Model predictions with calibrated confidence
    - Ground truth labels
    - Various failure scenarios
    """
    
    def __init__(
        self,
        config: Optional[SyntheticDataConfig] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config or SyntheticDataConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        
        # Track generation state
        self._batch_count = 0
        self._drift_offset = 0.0
    
    def reset(self) -> None:
        """Reset generator state."""
        self._batch_count = 0
        self._drift_offset = 0.0
        self.rng = np.random.default_rng(self.config.random_seed)
    
    # ========== Basic Generation ==========
    
    def generate_features(
        self,
        n_samples: Optional[int] = None,
        pattern: DataPattern = DataPattern.NORMAL,
    ) -> NDArray[np.floating]:
        """
        Generate synthetic feature data.
        
        Args:
            n_samples: Number of samples
            pattern: Data pattern to generate
            
        Returns:
            Feature array (n_samples, n_features)
        """
        n_samples = n_samples or self.config.n_samples
        n_features = self.config.n_features
        
        # Base features from standard normal
        X = self.rng.standard_normal((n_samples, n_features))
        
        # Apply pattern
        if pattern == DataPattern.GRADUAL_DRIFT:
            drift = np.linspace(0, self.config.drift_rate * n_samples, n_samples)
            X += drift.reshape(-1, 1)
            self._drift_offset += self.config.drift_rate * n_samples
        
        elif pattern == DataPattern.SUDDEN_DRIFT:
            # Shift halfway through
            mid = n_samples // 2
            X[mid:] += 2.0
        
        elif pattern == DataPattern.INCREASING_NOISE:
            noise_scale = np.linspace(1, 3, n_samples)
            X *= noise_scale.reshape(-1, 1)
        
        elif pattern == DataPattern.PERIODIC:
            t = np.linspace(0, 4 * np.pi, n_samples)
            X += np.sin(t).reshape(-1, 1) * 0.5
        
        elif pattern == DataPattern.DEGRADING:
            # Progressive degradation
            degradation = np.linspace(0, 1, n_samples)
            noise = self.rng.standard_normal((n_samples, n_features))
            X = X * (1 - degradation.reshape(-1, 1)) + noise * degradation.reshape(-1, 1) * 2
        
        elif pattern == DataPattern.MIXED:
            # Combine multiple patterns
            X = self._apply_mixed_patterns(X)
        
        self._batch_count += 1
        return X.astype(np.float64)
    
    def _apply_mixed_patterns(
        self, X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply mixed patterns to data."""
        n_samples = X.shape[0]
        
        # Apply drift to first half of features
        half_f = self.config.n_features // 2
        drift = np.linspace(0, self.config.drift_rate * n_samples, n_samples)
        X[:, :half_f] += drift.reshape(-1, 1)
        
        # Add periodic component
        t = np.linspace(0, 2 * np.pi, n_samples)
        X += np.sin(t).reshape(-1, 1) * 0.3
        
        return X
    
    def generate_labels(
        self,
        X: NDArray[np.floating],
        noise_rate: Optional[float] = None,
    ) -> NDArray[np.int_]:
        """
        Generate labels based on features.
        
        Args:
            X: Feature data
            noise_rate: Label noise rate (default from config)
            
        Returns:
            Label array
        """
        noise_rate = noise_rate if noise_rate is not None else self.config.noise_level
        n_samples = X.shape[0]
        
        # Simple linear decision boundary
        scores = np.sum(X[:, :3], axis=1)  # Use first 3 features
        labels = (scores > 0).astype(np.int_)
        
        # Add label noise
        noise_mask = self.rng.random(n_samples) < noise_rate
        labels[noise_mask] = 1 - labels[noise_mask]
        
        return labels
    
    def generate_predictions(
        self,
        y_true: NDArray[np.int_],
        accuracy: float = 0.85,
    ) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
        """
        Generate model predictions with controlled accuracy.
        
        Args:
            y_true: Ground truth labels
            accuracy: Target accuracy
            
        Returns:
            Tuple of (predictions, confidences)
        """
        n_samples = len(y_true)
        
        # Generate predictions with target accuracy
        predictions = y_true.copy()
        n_errors = int(n_samples * (1 - accuracy))
        
        error_indices = self.rng.choice(n_samples, n_errors, replace=False)
        predictions[error_indices] = 1 - predictions[error_indices]
        
        # Generate confidences
        # Correct predictions: higher confidence
        # Incorrect predictions: mixed confidence
        confidences = np.zeros(n_samples, dtype=np.float64)
        
        correct_mask = predictions == y_true
        confidences[correct_mask] = 0.7 + self.rng.random(np.sum(correct_mask)) * 0.25
        confidences[~correct_mask] = 0.3 + self.rng.random(np.sum(~correct_mask)) * 0.5
        
        return predictions, confidences
    
    # ========== Scenario Generation ==========
    
    def generate_normal_scenario(
        self,
        n_batches: int = 10,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate a normal operation scenario.
        
        Args:
            n_batches: Number of batches
            batch_size: Samples per batch
            
        Returns:
            List of batch dictionaries
        """
        batches = []
        
        for _ in range(n_batches):
            X = self.generate_features(batch_size, DataPattern.NORMAL)
            y_true = self.generate_labels(X)
            y_pred, conf = self.generate_predictions(y_true, accuracy=0.90)
            
            batches.append({
                "X": X,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": conf,
            })
        
        return batches
    
    def generate_drift_scenario(
        self,
        n_batches: int = 20,
        batch_size: int = 100,
        drift_type: str = "gradual",
    ) -> List[Dict[str, Any]]:
        """
        Generate a drift scenario.
        
        Args:
            n_batches: Number of batches
            batch_size: Samples per batch
            drift_type: "gradual" or "sudden"
            
        Returns:
            List of batch dictionaries
        """
        batches = []
        pattern = DataPattern.GRADUAL_DRIFT if drift_type == "gradual" else DataPattern.SUDDEN_DRIFT
        
        # Initial accuracy degrades over time
        initial_accuracy = 0.90
        
        for i in range(n_batches):
            X = self.generate_features(batch_size, pattern)
            y_true = self.generate_labels(X)
            
            # Accuracy degrades with drift
            if drift_type == "gradual":
                accuracy = initial_accuracy - (i / n_batches) * 0.3
            else:
                accuracy = initial_accuracy if i < n_batches // 2 else initial_accuracy - 0.25
            
            y_pred, conf = self.generate_predictions(y_true, accuracy=max(0.5, accuracy))
            
            batches.append({
                "X": X,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": conf,
                "expected_accuracy": accuracy,
            })
        
        return batches
    
    def generate_failure_scenario(
        self,
        n_batches: int = 15,
        batch_size: int = 100,
        failure_point: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate a failure scenario.
        
        Args:
            n_batches: Number of batches
            batch_size: Samples per batch
            failure_point: Batch where failure occurs
            
        Returns:
            List of batch dictionaries
        """
        batches = []
        
        for i in range(n_batches):
            if i < failure_point:
                X = self.generate_features(batch_size, DataPattern.NORMAL)
                y_true = self.generate_labels(X)
                y_pred, conf = self.generate_predictions(y_true, accuracy=0.88)
            else:
                # Post-failure: sudden degradation
                X = self.generate_features(batch_size, DataPattern.SUDDEN_DRIFT)
                y_true = self.generate_labels(X, noise_rate=0.3)
                
                # Model becomes overconfident while wrong
                accuracy = 0.55 - (i - failure_point) * 0.03
                y_pred, conf = self.generate_predictions(y_true, accuracy=max(0.4, accuracy))
                
                # Overconfidence pattern
                conf = conf * 1.2
                conf = np.clip(conf, 0, 0.99)
            
            batches.append({
                "X": X,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": conf,
                "is_failure_region": i >= failure_point,
            })
        
        return batches
    
    def generate_recovery_scenario(
        self,
        n_batches: int = 25,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate a degradation and recovery scenario.
        
        Args:
            n_batches: Number of batches
            batch_size: Samples per batch
            
        Returns:
            List of batch dictionaries
        """
        batches = []
        
        # Phase 1: Normal (0-8)
        # Phase 2: Degradation (9-16)
        # Phase 3: Recovery (17-25)
        
        for i in range(n_batches):
            if i < 8:
                # Normal phase
                pattern = DataPattern.NORMAL
                accuracy = 0.90
            elif i < 17:
                # Degradation phase
                pattern = DataPattern.DEGRADING
                progress = (i - 8) / 8
                accuracy = 0.90 - progress * 0.35
            else:
                # Recovery phase
                pattern = DataPattern.NORMAL
                progress = (i - 17) / 8
                accuracy = 0.55 + progress * 0.30
            
            X = self.generate_features(batch_size, pattern)
            y_true = self.generate_labels(X)
            y_pred, conf = self.generate_predictions(y_true, accuracy=max(0.5, accuracy))
            
            phase = "normal" if i < 8 else "degradation" if i < 17 else "recovery"
            
            batches.append({
                "X": X,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": conf,
                "phase": phase,
                "expected_accuracy": accuracy,
            })
        
        return batches
    
    # ========== Utilities ==========
    
    def generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        return [f"feature_{i}" for i in range(self.config.n_features)]
    
    def generate_stream(
        self,
        pattern: DataPattern = DataPattern.NORMAL,
        batch_size: int = 50,
    ):
        """
        Generate infinite stream of data batches.
        
        Args:
            pattern: Data pattern
            batch_size: Samples per batch
            
        Yields:
            Batch dictionaries
        """
        while True:
            X = self.generate_features(batch_size, pattern)
            y_true = self.generate_labels(X)
            y_pred, conf = self.generate_predictions(y_true, accuracy=0.85)
            
            yield {
                "X": X,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": conf,
                "batch_id": self._batch_count,
            }
