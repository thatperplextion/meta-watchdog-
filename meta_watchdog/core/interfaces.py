"""
Core Interfaces and Protocols for Meta-Watchdog System

This module defines the abstract contracts that all components must follow,
ensuring a consistent and pluggable architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PredictionModel(Protocol):
    """
    Protocol for any prediction model that can be monitored by Meta-Watchdog.
    
    Any ML model can implement this interface to become self-aware.
    The system is completely dataset-agnostic - it only cares about:
    1. Making predictions
    2. Providing confidence scores
    """
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Make predictions on input data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        ...
    
    def predict_with_confidence(
        self, X: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Make predictions with associated confidence scores.
        
        This is the KEY method for self-awareness. The model must be able
        to express uncertainty about its predictions.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Tuple of:
                - predictions: shape (n_samples,) or (n_samples, n_outputs)
                - confidence: shape (n_samples,) values in [0, 1]
        """
        ...
    
    @property
    def model_type(self) -> str:
        """Return the type of model: 'classification' or 'regression'"""
        ...
    
    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained"""
        ...


class Monitor(ABC):
    """
    Abstract base class for monitoring components.
    
    Monitors observe the system state and record relevant metrics
    without interfering with the prediction pipeline.
    """
    
    @abstractmethod
    def observe(self, data: Dict[str, Any]) -> None:
        """
        Record an observation from the system.
        
        Args:
            data: Dictionary containing observation data
        """
        pass
    
    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of what's being monitored.
        
        Returns:
            Dictionary with current monitoring state
        """
        pass
    
    @abstractmethod
    def get_history(self, window_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical observations.
        
        Args:
            window_size: Number of recent observations to return.
                        If None, returns all history.
                        
        Returns:
            List of historical observation dictionaries
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Clear all stored observations and reset state."""
        pass


class Analyzer(ABC):
    """
    Abstract base class for analysis components.
    
    Analyzers take monitoring data and extract meaningful insights
    about system health, trends, and potential issues.
    """
    
    @abstractmethod
    def analyze(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis on monitoring data.
        
        Args:
            monitoring_data: Data from monitoring components
            
        Returns:
            Analysis results as a dictionary
        """
        pass
    
    @abstractmethod
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent analyses.
        
        Returns:
            Summary dictionary with key findings
        """
        pass


class Recommender(ABC):
    """
    Abstract base class for recommendation components.
    
    Recommenders take analysis results and generate actionable
    recommendations for maintaining system health.
    """
    
    @abstractmethod
    def generate_recommendations(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on analysis.
        
        Args:
            analysis: Analysis results from Analyzer components
            
        Returns:
            List of recommendation dictionaries
        """
        pass
    
    @abstractmethod
    def prioritize_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prioritize recommendations by urgency and impact.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Sorted list with highest priority first
        """
        pass


class Explainer(ABC):
    """
    Abstract base class for explainability components.
    
    Explainers transform technical findings into human-readable
    explanations that answer: What? Why? What should be done?
    """
    
    @abstractmethod
    def explain(
        self, 
        findings: Dict[str, Any],
        detail_level: str = "standard"
    ) -> str:
        """
        Generate a human-readable explanation of findings.
        
        Args:
            findings: Technical findings to explain
            detail_level: One of "brief", "standard", "detailed"
            
        Returns:
            Human-readable explanation string
        """
        pass
    
    @abstractmethod
    def explain_prediction(
        self,
        prediction: Any,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Explain a specific prediction.
        
        Args:
            prediction: The prediction value
            confidence: Confidence score for the prediction
            context: Optional additional context
            
        Returns:
            Explanation of the prediction
        """
        pass
    
    @abstractmethod
    def explain_failure_risk(
        self,
        failure_probability: float,
        time_to_failure: Optional[int],
        causes: List[Dict[str, Any]]
    ) -> str:
        """
        Explain the current failure risk assessment.
        
        Args:
            failure_probability: Probability of failure
            time_to_failure: Estimated predictions until failure
            causes: List of identified causes
            
        Returns:
            Human-readable risk explanation
        """
        pass


class Simulator(ABC):
    """
    Abstract base class for simulation components.
    
    Simulators generate counterfactual scenarios to stress-test
    the system and identify potential vulnerabilities.
    """
    
    @abstractmethod
    def generate_scenario(
        self,
        base_data: NDArray[np.floating],
        scenario_type: str,
        severity: float = 0.5
    ) -> NDArray[np.floating]:
        """
        Generate a counterfactual data scenario.
        
        Args:
            base_data: Original data to transform
            scenario_type: Type of scenario to generate
            severity: Severity of the scenario (0-1)
            
        Returns:
            Transformed data representing the scenario
        """
        pass
    
    @abstractmethod
    def get_available_scenarios(self) -> List[str]:
        """
        Get list of available scenario types.
        
        Returns:
            List of scenario type names
        """
        pass
    
    @abstractmethod
    def run_stress_test(
        self,
        model: PredictionModel,
        base_data: NDArray[np.floating],
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run stress tests against a model.
        
        Args:
            model: Model to test
            base_data: Data to use as baseline
            scenarios: Specific scenarios to run (or all if None)
            
        Returns:
            Stress test results
        """
        pass


class SensitivityMapper(ABC):
    """
    Abstract base class for sensitivity analysis.
    
    Maps which features and conditions are most likely to cause
    model failure under various scenarios.
    """
    
    @abstractmethod
    def compute_sensitivity(
        self,
        model: PredictionModel,
        data: NDArray[np.floating],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute sensitivity of model to each feature.
        
        Args:
            model: Model to analyze
            data: Sample data for analysis
            feature_names: Optional names for features
            
        Returns:
            Dictionary mapping feature names to sensitivity scores
        """
        pass
    
    @abstractmethod
    def generate_sensitivity_map(
        self,
        model: PredictionModel,
        data: NDArray[np.floating],
        scenarios: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate a full sensitivity map across scenarios.
        
        Args:
            model: Model to analyze
            data: Sample data for analysis
            scenarios: Scenarios to test
            
        Returns:
            Nested dict: scenario -> feature -> sensitivity
        """
        pass
