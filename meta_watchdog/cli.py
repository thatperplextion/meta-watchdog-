"""
Meta-Watchdog Command Line Interface

Provides CLI access to Meta-Watchdog functionality.
"""

import argparse
import sys
import json
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="meta-watchdog",
        description="Self-Aware Machine Learning System - Monitor, predict, and explain model failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  meta-watchdog status                    Show current system status
  meta-watchdog analyze --deep            Run deep analysis
  meta-watchdog dashboard                 Launch terminal dashboard
  meta-watchdog demo                      Run demonstration scenario
  meta-watchdog config --show             Show current configuration
        """
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed status"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis")
    analyze_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep analysis"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for analysis results"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch dashboard")
    dashboard_parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=5,
        help="Refresh interval in seconds"
    )
    dashboard_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    demo_parser.add_argument(
        "--scenario",
        choices=["normal", "drift", "failure", "recovery"],
        default="normal",
        help="Scenario to demonstrate"
    )
    demo_parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of batches to process"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--generate",
        type=str,
        help="Generate default config to specified path"
    )
    
    return parser


def cmd_status(args: argparse.Namespace) -> int:
    """Handle status command."""
    from meta_watchdog.orchestrator import MetaWatchdogOrchestrator
    from meta_watchdog.core.base_model import BaseModel
    import numpy as np
    
    # Create minimal model for status
    class DummyModel(BaseModel):
        def __init__(self):
            super().__init__("dummy", "1.0")
        def _predict_impl(self, X):
            return (np.sum(X, axis=1) > 0).astype(int)
        def _predict_proba_impl(self, X):
            return np.column_stack([0.5 * np.ones(len(X)), 0.5 * np.ones(len(X))])
    
    orchestrator = MetaWatchdogOrchestrator(model=DummyModel())
    status = orchestrator.get_quick_status()
    
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print("META-WATCHDOG STATUS")
        print("=" * 40)
        print(f"Status:            {status['status'].upper()}")
        print(f"Reliability Score: {status['reliability_score']:.1f}/100")
        print(f"Failure Risk:      {status['failure_probability']:.1%}")
        print(f"Active Alerts:     {status['active_alerts']}")
        print(f"Mode:              {status['mode']}")
        print(f"Observations:      {status['observations']}")
    
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle analyze command."""
    print("Analysis command - would run analysis on configured model")
    print("Use --deep for comprehensive analysis")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Handle dashboard command."""
    from meta_watchdog.orchestrator import MetaWatchdogOrchestrator
    from meta_watchdog.dashboard import DashboardDataProvider, TerminalDashboard
    from meta_watchdog.testing import SyntheticDataGenerator, SyntheticDataConfig
    from meta_watchdog.core.base_model import BaseModel
    import numpy as np
    import time
    
    class DemoModel(BaseModel):
        def __init__(self):
            super().__init__("demo", "1.0")
        def _predict_impl(self, X):
            return (np.sum(X, axis=1) > 0).astype(int)
        def _predict_proba_impl(self, X):
            scores = 1 / (1 + np.exp(-np.sum(X, axis=1) * 0.5))
            return np.column_stack([1 - scores, scores])
    
    orchestrator = MetaWatchdogOrchestrator(model=DemoModel())
    dashboard_provider = DashboardDataProvider(orchestrator)
    dashboard = TerminalDashboard(
        dashboard_provider,
        use_colors=not args.no_color,
    )
    
    # Generate some data
    gen = SyntheticDataGenerator(SyntheticDataConfig(n_features=5, random_seed=42))
    
    print("\nMeta-Watchdog Dashboard")
    print("Press Ctrl+C to exit\n")
    
    try:
        batch_idx = 0
        while True:
            # Generate and observe data
            X = gen.generate_features(50)
            y_true = gen.generate_labels(X)
            y_pred, conf = gen.generate_predictions(y_true)
            
            orchestrator.observe(X, y_true, y_pred, conf)
            batch_idx += 1
            
            # Clear screen and render dashboard
            print("\033[2J\033[H")  # Clear screen
            print(dashboard.render())
            print(f"\nBatches processed: {batch_idx}")
            
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nDashboard closed.")
    
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Handle demo command."""
    from meta_watchdog.orchestrator import MetaWatchdogOrchestrator
    from meta_watchdog.testing import SyntheticDataGenerator, SyntheticDataConfig
    from meta_watchdog.core.base_model import BaseModel
    import numpy as np
    
    class DemoModel(BaseModel):
        def __init__(self):
            super().__init__("demo", "1.0")
        def _predict_impl(self, X):
            return (np.sum(X, axis=1) > 0).astype(int)
        def _predict_proba_impl(self, X):
            scores = 1 / (1 + np.exp(-np.sum(X, axis=1) * 0.5))
            return np.column_stack([1 - scores, scores])
    
    print(f"Running {args.scenario} scenario demo with {args.batches} batches...")
    print("=" * 50)
    
    orchestrator = MetaWatchdogOrchestrator(model=DemoModel())
    gen = SyntheticDataGenerator(SyntheticDataConfig(n_features=5, random_seed=42))
    
    # Select scenario
    if args.scenario == "normal":
        batches = gen.generate_normal_scenario(args.batches, 50)
    elif args.scenario == "drift":
        batches = gen.generate_drift_scenario(args.batches, 50)
    elif args.scenario == "failure":
        batches = gen.generate_failure_scenario(args.batches, 50, args.batches // 2)
    else:  # recovery
        batches = gen.generate_recovery_scenario(args.batches, 50)
    
    for i, batch in enumerate(batches):
        result = orchestrator.observe(
            batch["X"], batch["y_true"], batch["y_pred"], batch["confidence"]
        )
        
        score = result["reliability"].score
        risk = result["failure_prediction"].probability
        
        # Visual indicator
        if score >= 70:
            indicator = "✓"
        elif score >= 50:
            indicator = "⚠"
        else:
            indicator = "✗"
        
        print(f"Batch {i+1:3d}: Reliability={score:5.1f} | Risk={risk:5.1%} {indicator}")
    
    # Final status
    print("=" * 50)
    status = orchestrator.get_quick_status()
    print(f"Final Status: {status['status'].upper()}")
    print(f"Total Alerts: {status['active_alerts']}")
    
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Handle config command."""
    if args.show:
        from config.config_manager import ConfigManager
        
        manager = ConfigManager()
        config_dict = manager.to_dict()
        
        print(json.dumps(config_dict, indent=2))
    
    elif args.generate:
        import shutil
        from pathlib import Path
        
        default_config = Path(__file__).parent.parent / "config" / "default_config.yaml"
        if default_config.exists():
            shutil.copy(default_config, args.generate)
            print(f"Configuration generated at: {args.generate}")
        else:
            print("Default configuration not found")
            return 1
    
    return 0


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Route to command handler
    handlers = {
        "status": cmd_status,
        "analyze": cmd_analyze,
        "dashboard": cmd_dashboard,
        "demo": cmd_demo,
        "config": cmd_config,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
