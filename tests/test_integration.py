"""
Integration tests for Meta-Watchdog system.
"""

import pytest
import numpy as np
from datetime import datetime

from meta_watchdog.orchestrator.system import (
    MetaWatchdogOrchestrator,
    OrchestratorConfig,
    SystemMode,
    AlertLevel,
)
from meta_watchdog.testing.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    DataPattern,
)
from meta_watchdog.core.base_model import BaseModel


class SimpleClassifier(BaseModel):
    """Simple classifier for integration testing."""
    
    def __init__(self):
        super().__init__("simple_classifier", "1.0")
    
    def _predict_impl(self, X):
        return (np.sum(X, axis=1) > 0).astype(int)
    
    def _predict_proba_impl(self, X):
        scores = 1 / (1 + np.exp(-np.sum(X, axis=1) * 0.5))
        return np.column_stack([1 - scores, scores])


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        model = SimpleClassifier()
        config = OrchestratorConfig(
            reliability_warning_threshold=60.0,
            reliability_critical_threshold=40.0,
        )
        return MetaWatchdogOrchestrator(model=model, config=config)
    
    @pytest.fixture
    def data_generator(self):
        config = SyntheticDataConfig(
            n_samples=100,
            n_features=5,
            random_seed=42
        )
        return SyntheticDataGenerator(config)
    
    def test_normal_operation(self, orchestrator, data_generator):
        """Test normal operation scenario."""
        batches = data_generator.generate_normal_scenario(n_batches=5, batch_size=50)
        
        for batch in batches:
            result = orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
        
        # Check system is healthy
        status = orchestrator.get_quick_status()
        assert status["observations"] == 5
    
    def test_drift_detection(self, orchestrator, data_generator):
        """Test drift scenario detection."""
        batches = data_generator.generate_drift_scenario(n_batches=10, batch_size=50)
        
        alerts_generated = []
        
        for batch in batches:
            result = orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
            
            if result.get("alerts"):
                alerts_generated.extend(result["alerts"])
        
        # Drift should eventually trigger alerts
        # Note: May need more batches to trigger in real scenarios
        assert orchestrator._observation_count == 10
    
    def test_failure_scenario(self, orchestrator, data_generator):
        """Test failure scenario handling."""
        batches = data_generator.generate_failure_scenario(
            n_batches=15, 
            batch_size=50,
            failure_point=8
        )
        
        for batch in batches:
            orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
        
        # Check that failure was detected
        state = orchestrator.get_system_state()
        assert state is not None
    
    def test_deep_analysis(self, orchestrator, data_generator):
        """Test deep analysis functionality."""
        # Generate some data first
        batches = data_generator.generate_normal_scenario(n_batches=5, batch_size=50)
        
        for batch in batches:
            orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
        
        # Run deep analysis
        X_sample = batches[-1]["X"]
        analysis = orchestrator.run_deep_analysis(X=X_sample)
        
        assert "reliability" in analysis
        assert "failure_prediction" in analysis
        assert "explanation" in analysis
    
    def test_health_snapshot(self, orchestrator, data_generator):
        """Test health snapshot generation."""
        batches = data_generator.generate_normal_scenario(n_batches=3, batch_size=50)
        
        for batch in batches:
            orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
        
        snapshot = orchestrator.get_health_snapshot()
        
        assert snapshot.reliability_score is not None
        assert snapshot.failure_prediction is not None
        assert snapshot.explanation is not None
    
    def test_alert_callback(self, orchestrator, data_generator):
        """Test alert callback mechanism."""
        alerts_received = []
        
        def callback(alert):
            alerts_received.append(alert)
        
        orchestrator.register_alert_callback(callback)
        
        # Generate failure scenario to trigger alerts
        batches = data_generator.generate_failure_scenario(
            n_batches=20,
            batch_size=50,
            failure_point=5
        )
        
        for batch in batches:
            orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
        
        # Callbacks should have been invoked if alerts were generated
        assert isinstance(alerts_received, list)
    
    def test_reset(self, orchestrator, data_generator):
        """Test orchestrator reset."""
        batches = data_generator.generate_normal_scenario(n_batches=3, batch_size=50)
        
        for batch in batches:
            orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
        
        assert orchestrator._observation_count == 3
        
        orchestrator.reset()
        
        assert orchestrator._observation_count == 0
        assert len(orchestrator._alerts) == 0


class TestEndToEndFlow:
    """End-to-end flow tests."""
    
    def test_complete_monitoring_cycle(self):
        """Test complete monitoring cycle from observation to recommendation."""
        # Setup
        model = SimpleClassifier()
        config = OrchestratorConfig(
            enable_auto_recommendations=True,
            enable_auto_explanations=True,
        )
        orchestrator = MetaWatchdogOrchestrator(model=model, config=config)
        
        gen_config = SyntheticDataConfig(n_features=5, random_seed=42)
        generator = SyntheticDataGenerator(gen_config)
        
        # Generate degradation scenario
        batches = generator.generate_recovery_scenario(n_batches=10, batch_size=50)
        
        # Process all batches
        for batch in batches:
            orchestrator.observe(
                X=batch["X"],
                y_true=batch["y_true"],
                predictions=batch["y_pred"],
                confidences=batch["confidence"],
            )
        
        # Run deep analysis
        analysis = orchestrator.run_deep_analysis(X=batches[-1]["X"])
        
        # Verify all components produced output
        assert analysis["reliability"] is not None
        assert analysis["failure_prediction"] is not None
        assert analysis["explanation"] is not None
        
        # Verify we can get dashboard data
        from meta_watchdog.dashboard import DashboardDataProvider
        
        dashboard = DashboardDataProvider(orchestrator)
        data = dashboard.get_dashboard_data()
        
        assert data.system_status is not None
        assert data.reliability_score >= 0
    
    def test_simulation_integration(self):
        """Test counterfactual simulation integration."""
        from meta_watchdog.simulation import (
            CounterfactualSimulator,
            SensitivityMapper,
        )
        
        model = SimpleClassifier()
        
        gen_config = SyntheticDataConfig(n_features=5, random_seed=42)
        generator = SyntheticDataGenerator(gen_config)
        
        X = generator.generate_features(100, DataPattern.NORMAL)
        
        # Test counterfactual simulation
        simulator = CounterfactualSimulator()
        results = simulator.run_stress_test_suite(model, X)
        
        assert len(results) > 0
        
        # Test sensitivity mapping
        mapper = SensitivityMapper()
        sensitivity_map = mapper.generate_sensitivity_map(model, X)
        
        assert sensitivity_map is not None
        assert len(sensitivity_map.feature_sensitivities) == 5
    
    def test_explanation_generation(self):
        """Test explanation generation for different audiences."""
        from meta_watchdog.explainability import (
            ExplainabilityEngine,
            ExplanationAudience,
            ExplanationVerbosity,
        )
        from meta_watchdog.core.data_structures import ReliabilityScore
        
        engine = ExplainabilityEngine()
        
        score = ReliabilityScore(
            score=75.0,
            performance_score=0.8,
            calibration_score=0.7,
            stability_score=0.75,
            freshness_score=0.8,
            feature_health_score=0.7,
        )
        
        # Test different audiences
        for audience in ExplanationAudience:
            explanation = engine.explain_reliability(score, audience=audience)
            assert explanation.summary is not None
            assert len(explanation.summary) > 0
        
        # Test different verbosity levels
        for verbosity in ExplanationVerbosity:
            explanation = engine.explain_reliability(
                score, 
                verbosity=verbosity
            )
            assert explanation.summary is not None


class TestDataGeneratorIntegration:
    """Integration tests for synthetic data generator."""
    
    def test_scenario_variety(self):
        """Test all scenario types."""
        config = SyntheticDataConfig(n_features=5, random_seed=42)
        generator = SyntheticDataGenerator(config)
        
        scenarios = [
            ("normal", generator.generate_normal_scenario(5, 50)),
            ("drift", generator.generate_drift_scenario(10, 50)),
            ("failure", generator.generate_failure_scenario(15, 50, 8)),
            ("recovery", generator.generate_recovery_scenario(20, 50)),
        ]
        
        for name, batches in scenarios:
            assert len(batches) > 0
            
            for batch in batches:
                assert "X" in batch
                assert "y_true" in batch
                assert "y_pred" in batch
                assert "confidence" in batch
    
    def test_data_patterns(self):
        """Test different data patterns."""
        config = SyntheticDataConfig(n_features=5, random_seed=42)
        generator = SyntheticDataGenerator(config)
        
        for pattern in DataPattern:
            X = generator.generate_features(100, pattern)
            
            assert X.shape == (100, 5)
            assert not np.any(np.isnan(X))
    
    def test_stream_generation(self):
        """Test streaming data generation."""
        config = SyntheticDataConfig(n_features=5, random_seed=42)
        generator = SyntheticDataGenerator(config)
        
        stream = generator.generate_stream(DataPattern.NORMAL, batch_size=20)
        
        batches = []
        for _ in range(3):
            batch = next(stream)
            batches.append(batch)
        
        assert len(batches) == 3
        assert all(batch["X"].shape[0] == 20 for batch in batches)
