"""
Scenario Generators for Counterfactual Simulation

Generates various "what-if" data scenarios to stress-test models.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from enum import Enum

from meta_watchdog.core.data_structures import ScenarioType


@dataclass
class ScenarioConfig:
    """Configuration for a scenario."""
    scenario_type: ScenarioType
    severity: float = 0.5  # 0-1
    affected_features: Optional[List[int]] = None  # None = all features
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.severity <= 1:
            raise ValueError("Severity must be between 0 and 1")


@dataclass
class GeneratedScenario:
    """Result of scenario generation."""
    original_data: NDArray[np.floating]
    modified_data: NDArray[np.floating]
    scenario_type: ScenarioType
    severity: float
    affected_features: List[int]
    modifications: Dict[str, Any]
    description: str


class ScenarioGenerator:
    """
    Generates counterfactual data scenarios for model stress testing.
    
    Each scenario simulates a potential real-world data shift that
    could cause the model to fail. By testing against these scenarios,
    we can identify vulnerabilities before they occur in production.
    
    Available Scenarios:
    - Feature Drift: Gradual shift in feature distributions
    - Noise Injection: Add noise to simulate data quality issues
    - Trend Shift: Change underlying trends in the data
    - Feature Decay: Reduce relevance of certain features
    - Outlier Storm: Inject extreme values
    - Missing Values: Simulate missing data patterns
    - Correlation Breakdown: Change feature correlations
    - Scale Shift: Change scale/magnitude of features
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the scenario generator.
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)
        
        # Register scenario generators
        self._generators: Dict[ScenarioType, Callable] = {
            ScenarioType.FEATURE_DRIFT: self._generate_feature_drift,
            ScenarioType.NOISE_INJECTION: self._generate_noise_injection,
            ScenarioType.TREND_SHIFT: self._generate_trend_shift,
            ScenarioType.FEATURE_DECAY: self._generate_feature_decay,
            ScenarioType.OUTLIER_STORM: self._generate_outlier_storm,
            ScenarioType.MISSING_VALUES: self._generate_missing_values,
            ScenarioType.CORRELATION_BREAKDOWN: self._generate_correlation_breakdown,
            ScenarioType.SCALE_SHIFT: self._generate_scale_shift,
        }
    
    def generate(
        self,
        data: NDArray[np.floating],
        config: ScenarioConfig
    ) -> GeneratedScenario:
        """
        Generate a counterfactual scenario.
        
        Args:
            data: Original data (n_samples, n_features)
            config: Scenario configuration
            
        Returns:
            GeneratedScenario with modified data
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        generator = self._generators.get(config.scenario_type)
        if generator is None:
            raise ValueError(f"Unknown scenario type: {config.scenario_type}")
        
        # Determine affected features
        n_features = data.shape[1]
        if config.affected_features is None:
            affected = list(range(n_features))
        else:
            affected = [f for f in config.affected_features if 0 <= f < n_features]
        
        # Generate scenario
        modified, modifications = generator(
            data, config.severity, affected, config.parameters
        )
        
        return GeneratedScenario(
            original_data=data,
            modified_data=modified,
            scenario_type=config.scenario_type,
            severity=config.severity,
            affected_features=affected,
            modifications=modifications,
            description=self._get_scenario_description(config.scenario_type, config.severity)
        )
    
    def generate_multiple(
        self,
        data: NDArray[np.floating],
        scenarios: List[ScenarioConfig]
    ) -> List[GeneratedScenario]:
        """Generate multiple scenarios."""
        return [self.generate(data, config) for config in scenarios]
    
    def generate_severity_sweep(
        self,
        data: NDArray[np.floating],
        scenario_type: ScenarioType,
        severities: Optional[List[float]] = None
    ) -> List[GeneratedScenario]:
        """
        Generate scenarios across a range of severities.
        
        Args:
            data: Original data
            scenario_type: Type of scenario
            severities: List of severity values (default: 0.1 to 1.0)
            
        Returns:
            List of scenarios at different severities
        """
        if severities is None:
            severities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        scenarios = []
        for severity in severities:
            config = ScenarioConfig(scenario_type=scenario_type, severity=severity)
            scenarios.append(self.generate(data, config))
        
        return scenarios
    
    # ========== Scenario Generators ==========
    
    def _generate_feature_drift(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Simulate gradual drift in feature distributions.
        
        This simulates real-world scenarios where input data
        gradually shifts away from training distribution.
        """
        modified = data.copy()
        
        # Drift parameters
        drift_type = params.get("drift_type", "mean_shift")
        
        modifications = {
            "drift_type": drift_type,
            "shifts": {}
        }
        
        for feat_idx in affected:
            feat_data = data[:, feat_idx]
            feat_std = np.std(feat_data)
            feat_mean = np.mean(feat_data)
            
            if drift_type == "mean_shift":
                # Shift the mean
                shift = severity * feat_std * 2  # Up to 2 std shift at max severity
                modified[:, feat_idx] = feat_data + shift
                modifications["shifts"][feat_idx] = {"mean_shift": float(shift)}
            
            elif drift_type == "variance_change":
                # Change variance
                scale = 1 + severity * 2  # Up to 3x variance at max severity
                modified[:, feat_idx] = feat_mean + (feat_data - feat_mean) * scale
                modifications["shifts"][feat_idx] = {"variance_scale": float(scale)}
            
            elif drift_type == "distribution_shift":
                # Shift distribution shape (add skewness)
                skew_factor = severity * 2
                modified[:, feat_idx] = feat_data + skew_factor * (feat_data - feat_mean) ** 2 / (feat_std + 1e-8)
                modifications["shifts"][feat_idx] = {"skew_factor": float(skew_factor)}
        
        return modified, modifications
    
    def _generate_noise_injection(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Inject noise to simulate data quality degradation.
        
        Simulates sensor noise, measurement errors, or
        data collection issues.
        """
        modified = data.copy()
        noise_type = params.get("noise_type", "gaussian")
        
        modifications = {"noise_type": noise_type, "noise_levels": {}}
        
        for feat_idx in affected:
            feat_std = np.std(data[:, feat_idx])
            noise_level = severity * feat_std  # Noise proportional to feature std
            
            if noise_type == "gaussian":
                noise = self.rng.normal(0, noise_level, data.shape[0])
            elif noise_type == "uniform":
                noise = self.rng.uniform(-noise_level * 2, noise_level * 2, data.shape[0])
            elif noise_type == "spike":
                # Occasional large spikes
                spike_mask = self.rng.random(data.shape[0]) < severity * 0.1
                noise = np.zeros(data.shape[0])
                noise[spike_mask] = self.rng.normal(0, feat_std * 3, np.sum(spike_mask))
            else:
                noise = self.rng.normal(0, noise_level, data.shape[0])
            
            modified[:, feat_idx] = data[:, feat_idx] + noise
            modifications["noise_levels"][feat_idx] = float(noise_level)
        
        return modified, modifications
    
    def _generate_trend_shift(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Introduce trend changes in the data.
        
        Simulates concept drift where relationships
        between features and targets change over time.
        """
        modified = data.copy()
        n_samples = data.shape[0]
        
        trend_type = params.get("trend_type", "linear")
        modifications = {"trend_type": trend_type, "trends": {}}
        
        # Create trend vector
        t = np.linspace(0, 1, n_samples)
        
        for feat_idx in affected:
            feat_range = np.ptp(data[:, feat_idx])  # Range
            
            if trend_type == "linear":
                trend = t * feat_range * severity
            elif trend_type == "exponential":
                trend = (np.exp(t * severity * 2) - 1) * feat_range * 0.5
            elif trend_type == "step":
                step_point = int(n_samples * (1 - severity))
                trend = np.zeros(n_samples)
                trend[step_point:] = feat_range * severity
            elif trend_type == "seasonal":
                periods = params.get("periods", 3)
                trend = np.sin(t * np.pi * 2 * periods) * feat_range * severity * 0.5
            else:
                trend = t * feat_range * severity
            
            modified[:, feat_idx] = data[:, feat_idx] + trend
            modifications["trends"][feat_idx] = {"max_trend": float(np.max(np.abs(trend)))}
        
        return modified, modifications
    
    def _generate_feature_decay(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Simulate decay in feature relevance.
        
        This simulates scenarios where previously informative
        features become less predictive over time.
        """
        modified = data.copy()
        decay_type = params.get("decay_type", "replace_with_noise")
        
        modifications = {"decay_type": decay_type, "decay_levels": {}}
        
        for feat_idx in affected:
            feat_data = data[:, feat_idx]
            
            if decay_type == "replace_with_noise":
                # Gradually replace signal with noise
                noise = self.rng.normal(np.mean(feat_data), np.std(feat_data), len(feat_data))
                modified[:, feat_idx] = (1 - severity) * feat_data + severity * noise
            
            elif decay_type == "compress_to_mean":
                # Compress values toward mean (reduce variance)
                feat_mean = np.mean(feat_data)
                modified[:, feat_idx] = feat_mean + (1 - severity) * (feat_data - feat_mean)
            
            elif decay_type == "randomize":
                # Shuffle values with probability proportional to severity
                shuffle_mask = self.rng.random(len(feat_data)) < severity
                shuffled = feat_data.copy()
                shuffled[shuffle_mask] = self.rng.permutation(feat_data[shuffle_mask])
                modified[:, feat_idx] = shuffled
            
            modifications["decay_levels"][feat_idx] = float(severity)
        
        return modified, modifications
    
    def _generate_outlier_storm(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Inject extreme outlier values.
        
        Simulates data corruption, edge cases, or
        adversarial inputs.
        """
        modified = data.copy()
        n_samples = data.shape[0]
        
        outlier_fraction = params.get("fraction", severity * 0.2)  # Up to 20% outliers
        outlier_magnitude = params.get("magnitude", 5)  # 5 std deviations
        
        modifications = {"outlier_fraction": outlier_fraction, "outliers": {}}
        
        for feat_idx in affected:
            feat_data = data[:, feat_idx]
            feat_mean = np.mean(feat_data)
            feat_std = np.std(feat_data)
            
            # Select outlier positions
            n_outliers = int(n_samples * outlier_fraction)
            outlier_indices = self.rng.choice(n_samples, n_outliers, replace=False)
            
            # Generate outlier values
            outlier_direction = self.rng.choice([-1, 1], n_outliers)
            outlier_values = feat_mean + outlier_direction * feat_std * outlier_magnitude * severity
            
            modified[outlier_indices, feat_idx] = outlier_values
            modifications["outliers"][feat_idx] = {
                "count": n_outliers,
                "magnitude": float(outlier_magnitude * severity)
            }
        
        return modified, modifications
    
    def _generate_missing_values(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Simulate missing data patterns.
        
        Uses NaN or specified fill value to represent missing data.
        """
        modified = data.copy()
        n_samples = data.shape[0]
        
        missing_fraction = severity * 0.3  # Up to 30% missing at max severity
        missing_type = params.get("type", "random")  # random, burst, structured
        fill_value = params.get("fill_value", np.nan)
        
        modifications = {"missing_fraction": missing_fraction, "missing_type": missing_type}
        
        for feat_idx in affected:
            if missing_type == "random":
                # Random missing values
                missing_mask = self.rng.random(n_samples) < missing_fraction
            
            elif missing_type == "burst":
                # Contiguous missing values
                burst_start = self.rng.integers(0, int(n_samples * 0.7))
                burst_length = int(n_samples * missing_fraction)
                missing_mask = np.zeros(n_samples, dtype=bool)
                missing_mask[burst_start:burst_start + burst_length] = True
            
            elif missing_type == "structured":
                # Missing based on feature value (e.g., low values more likely missing)
                feat_data = data[:, feat_idx]
                prob = 1 - (feat_data - feat_data.min()) / (feat_data.ptp() + 1e-8)
                missing_mask = self.rng.random(n_samples) < prob * missing_fraction * 2
            
            else:
                missing_mask = self.rng.random(n_samples) < missing_fraction
            
            modified[missing_mask, feat_idx] = fill_value
        
        return modified, modifications
    
    def _generate_correlation_breakdown(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Break correlations between features.
        
        Simulates scenarios where feature relationships
        that the model relies on no longer hold.
        """
        modified = data.copy()
        
        if len(affected) < 2:
            # Need at least 2 features for correlation
            return modified, {"message": "Need at least 2 features"}
        
        modifications = {"correlation_changes": []}
        
        # For each pair of affected features, break their correlation
        for i in range(len(affected) - 1):
            feat1, feat2 = affected[i], affected[i + 1]
            
            # Compute original correlation
            orig_corr = np.corrcoef(data[:, feat1], data[:, feat2])[0, 1]
            
            # Shuffle one feature to break correlation
            # Partial shuffle based on severity
            n_samples = data.shape[0]
            n_shuffle = int(n_samples * severity)
            shuffle_indices = self.rng.choice(n_samples, n_shuffle, replace=False)
            
            shuffled_values = modified[shuffle_indices, feat2].copy()
            self.rng.shuffle(shuffled_values)
            modified[shuffle_indices, feat2] = shuffled_values
            
            # Compute new correlation
            new_corr = np.corrcoef(modified[:, feat1], modified[:, feat2])[0, 1]
            
            modifications["correlation_changes"].append({
                "features": (feat1, feat2),
                "original_correlation": float(orig_corr),
                "new_correlation": float(new_corr),
            })
        
        return modified, modifications
    
    def _generate_scale_shift(
        self,
        data: NDArray[np.floating],
        severity: float,
        affected: List[int],
        params: Dict[str, Any]
    ) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
        """
        Shift the scale/magnitude of features.
        
        Simulates unit changes, sensor calibration issues,
        or normalization differences.
        """
        modified = data.copy()
        
        shift_type = params.get("shift_type", "multiply")  # multiply, offset, both
        modifications = {"shift_type": shift_type, "scales": {}}
        
        for feat_idx in affected:
            feat_data = data[:, feat_idx]
            
            if shift_type == "multiply":
                # Scale by a factor
                scale = 1 + (self.rng.random() * 2 - 1) * severity * 2  # 0.0x to 3.0x
                modified[:, feat_idx] = feat_data * scale
                modifications["scales"][feat_idx] = {"factor": float(scale)}
            
            elif shift_type == "offset":
                # Add offset
                offset = (self.rng.random() * 2 - 1) * np.std(feat_data) * severity * 3
                modified[:, feat_idx] = feat_data + offset
                modifications["scales"][feat_idx] = {"offset": float(offset)}
            
            elif shift_type == "both":
                scale = 1 + (self.rng.random() * 2 - 1) * severity * 2
                offset = (self.rng.random() * 2 - 1) * np.std(feat_data) * severity * 2
                modified[:, feat_idx] = feat_data * scale + offset
                modifications["scales"][feat_idx] = {"factor": float(scale), "offset": float(offset)}
        
        return modified, modifications
    
    # ========== Utilities ==========
    
    def _get_scenario_description(
        self, scenario_type: ScenarioType, severity: float
    ) -> str:
        """Get human-readable description of scenario."""
        descriptions = {
            ScenarioType.FEATURE_DRIFT: f"Feature distribution drift at {severity*100:.0f}% severity",
            ScenarioType.NOISE_INJECTION: f"Noise injection at {severity*100:.0f}% level",
            ScenarioType.TREND_SHIFT: f"Trend shift at {severity*100:.0f}% severity",
            ScenarioType.FEATURE_DECAY: f"Feature relevance decay at {severity*100:.0f}%",
            ScenarioType.OUTLIER_STORM: f"Outlier storm at {severity*100:.0f}% severity",
            ScenarioType.MISSING_VALUES: f"Missing values at {severity*100:.0f}% rate",
            ScenarioType.CORRELATION_BREAKDOWN: f"Correlation breakdown at {severity*100:.0f}%",
            ScenarioType.SCALE_SHIFT: f"Scale shift at {severity*100:.0f}% severity",
        }
        return descriptions.get(scenario_type, f"Unknown scenario at {severity*100:.0f}%")
    
    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenario types."""
        return [st.value for st in ScenarioType]
