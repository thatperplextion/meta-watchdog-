"""
Advanced Usage Example for Meta-Watchdog.

This example demonstrates the full capabilities of the Meta-Watchdog system
including custom models, alerts, plugins, and API integration.
"""

import numpy as np
import time
from typing import Dict, Any

# Import Meta-Watchdog components
from meta_watchdog import (
    MetaWatchdogOrchestrator,
    OrchestratorConfig,
    BaseModel,
    SyntheticDataGenerator,
)
from meta_watchdog.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertCategory,
    ConsoleAlertChannel,
    CallbackAlertChannel,
)
from meta_watchdog.benchmarks import Benchmark, Profiler, timed


class AdvancedClassifier(BaseModel):
    """
    Advanced classifier with built-in drift detection.
    
    This model simulates a production ML model with:
    - Confidence calibration
    - Performance tracking
    - Automatic degradation simulation
    """
    
    def __init__(self, base_accuracy: float = 0.95):
        super().__init__("advanced_classifier", "2.0")
        self.base_accuracy = base_accuracy
        self.predictions_made = 0
        self.drift_factor = 0.0
        self._weights = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the model."""
        n_features = X.shape[1]
        self._weights = np.random.randn(n_features) * 0.1
        
        # Simulate training
        return {
            "status": "trained",
            "n_samples": len(X),
            "n_features": n_features,
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with probability estimates."""
        if self._weights is None:
            self._weights = np.random.randn(X.shape[1]) * 0.1
        
        # Simulate predictions with degradation
        n_samples = len(X)
        
        # Base prediction quality degrades over time
        current_accuracy = self.base_accuracy - self.drift_factor
        
        # Generate predictions
        probas = np.random.rand(n_samples)
        
        # Adjust based on accuracy
        high_conf_mask = np.random.rand(n_samples) < current_accuracy
        probas[high_conf_mask] = np.clip(
            probas[high_conf_mask] + 0.3, 0.6, 0.99
        )
        
        self.predictions_made += n_samples
        
        return probas
    
    def inject_drift(self, amount: float = 0.05) -> None:
        """Simulate drift by reducing accuracy."""
        self.drift_factor = min(self.drift_factor + amount, 0.5)
        print(f"  [Drift injected] New drift factor: {self.drift_factor:.2f}")


def setup_alert_system() -> AlertManager:
    """Set up the alert management system."""
    manager = AlertManager()
    
    # Add console channel
    manager.add_channel("console", ConsoleAlertChannel(colored=True))
    
    # Add custom callback channel for integrations
    def slack_simulator(alert):
        print(f"  [Slack] Would send: {alert.title}")
        return True
    
    manager.add_channel("slack", CallbackAlertChannel(slack_simulator))
    
    # Add custom rules
    manager.add_rule(AlertRule(
        rule_id="custom_accuracy",
        name="Accuracy Drop Alert",
        condition=lambda ctx: ctx.get("accuracy", 1.0) < 0.85,
        severity=AlertSeverity.WARNING,
        category=AlertCategory.PERFORMANCE,
        title_template="Accuracy Below Threshold",
        message_template="Model accuracy dropped to {accuracy:.1%}",
        cooldown_seconds=30,
    ))
    
    manager.add_rule(AlertRule(
        rule_id="drift_detected",
        name="Drift Detection Alert",
        condition=lambda ctx: ctx.get("drift_score", 0) > 0.3,
        severity=AlertSeverity.ERROR,
        category=AlertCategory.DATA_QUALITY,
        title_template="Data Drift Detected",
        message_template="Drift score: {drift_score:.2f}",
        cooldown_seconds=60,
    ))
    
    return manager


@timed("process_batch")
def process_batch_with_timing(orchestrator, X, y):
    """Process a batch with timing instrumentation."""
    return orchestrator.process_batch(X, y)


def run_advanced_demo():
    """Run the advanced demonstration."""
    print("=" * 70)
    print("META-WATCHDOG: Advanced Usage Demonstration")
    print("=" * 70)
    print()
    
    # Initialize components
    print("[1] Initializing Advanced Classifier...")
    model = AdvancedClassifier(base_accuracy=0.95)
    
    print("[2] Setting up Meta-Watchdog Orchestrator...")
    config = OrchestratorConfig(
        reliability_threshold=70.0,
        failure_probability_threshold=0.4,
        enable_counterfactual=True,
        enable_root_cause=True,
    )
    orchestrator = MetaWatchdogOrchestrator(model=model, config=config)
    
    print("[3] Configuring Alert System...")
    alert_manager = setup_alert_system()
    
    print("[4] Initializing Synthetic Data Generator...")
    generator = SyntheticDataGenerator(n_features=15, seed=42)
    
    print("[5] Setting up Performance Profiler...")
    profiler = Profiler()
    
    print()
    print("-" * 70)
    print("PHASE 1: Normal Operation (20 batches)")
    print("-" * 70)
    
    benchmark = Benchmark(warmup_iterations=0)
    
    for i in range(20):
        with profiler.profile("batch_processing"):
            X, y = generator.generate_normal(n_samples=50)
            snapshot = process_batch_with_timing(orchestrator, X, y)
        
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1:2d}: Reliability={snapshot.reliability_score:.1f}, "
                  f"Failure P={snapshot.failure_probability:.2%}")
            
            # Evaluate alerts
            context = {
                "reliability_score": snapshot.reliability_score,
                "failure_probability": snapshot.failure_probability,
                "accuracy": 0.95 - (i * 0.005),  # Simulated
                "drift_score": 0.1,
            }
            alert_manager.evaluate_rules(context)
    
    print()
    print("-" * 70)
    print("PHASE 2: Gradual Drift Injection (15 batches)")
    print("-" * 70)
    
    for i in range(15):
        # Inject drift every 5 batches
        if i > 0 and i % 5 == 0:
            model.inject_drift(0.08)
        
        with profiler.profile("drift_processing"):
            X, y = generator.generate_drift(n_samples=50, drift_magnitude=0.1 * (i + 1))
            snapshot = orchestrator.process_batch(X, y)
        
        print(f"  Batch {i+1:2d}: Reliability={snapshot.reliability_score:.1f}, "
              f"Failure P={snapshot.failure_probability:.2%}")
        
        # Evaluate alerts with drift context
        context = {
            "reliability_score": snapshot.reliability_score,
            "failure_probability": snapshot.failure_probability,
            "accuracy": 0.95 - model.drift_factor,
            "drift_score": 0.1 + (i * 0.05),
        }
        alerts = alert_manager.evaluate_rules(context)
    
    print()
    print("-" * 70)
    print("PHASE 3: Analysis and Reporting")
    print("-" * 70)
    
    # Get full analysis
    print("\n[Analysis] Generating comprehensive system analysis...")
    analysis = orchestrator.get_full_analysis()
    
    print(f"\n  Final Reliability Score: {analysis.reliability_score:.1f}/100")
    print(f"  Failure Probability: {analysis.failure_probability:.2%}")
    print(f"  Root Causes Identified: {len(analysis.root_causes)}")
    
    if analysis.root_causes:
        print("\n  Top Root Causes:")
        for i, cause in enumerate(analysis.root_causes[:3], 1):
            print(f"    {i}. {cause.category.value}: {cause.confidence:.2%} confidence")
    
    if analysis.recommendations:
        print(f"\n  Recommendations ({len(analysis.recommendations)} total):")
        for i, rec in enumerate(analysis.recommendations[:3], 1):
            print(f"    {i}. [{rec.priority.value}] {rec.title}")
    
    # Alert statistics
    print("\n[Alerts] Alert System Statistics:")
    alert_stats = alert_manager.get_statistics()
    print(f"  Total Alerts: {alert_stats['total_alerts']}")
    print(f"  Active Alerts: {alert_stats['active_alerts']}")
    print(f"  By Severity: {alert_stats['by_severity']}")
    
    # Performance report
    print("\n[Performance] Profiling Report:")
    print(profiler.report())
    
    # Benchmark a single operation
    print("\n[Benchmark] Running prediction benchmark...")
    X_bench, _ = generator.generate_normal(n_samples=100)
    bench_result = benchmark.run(
        lambda: model.predict_proba(X_bench),
        name="predict_proba",
        iterations=50
    )
    print(f"  Mean prediction time: {bench_result.mean_time_ms:.3f}ms")
    print(f"  Throughput: {bench_result.throughput:.1f} ops/sec")
    
    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("This example demonstrated:")
    print("  ✓ Custom model implementation with BaseModel")
    print("  ✓ Alert system with custom rules and channels")
    print("  ✓ Drift injection and detection")
    print("  ✓ Root cause analysis")
    print("  ✓ Action recommendations")
    print("  ✓ Performance profiling and benchmarking")


if __name__ == "__main__":
    run_advanced_demo()
