"""
Configuration Management

Handles loading and managing Meta-Watchdog configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class ThresholdConfig:
    """Threshold configuration."""
    reliability_warning: float = 60.0
    reliability_critical: float = 40.0
    failure_alert: float = 0.5
    failure_critical: float = 0.8
    drift_threshold: float = 0.1
    calibration_error: float = 0.15


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    window_size: int = 100
    health_check_interval: int = 60
    deep_analysis_interval: int = 300


@dataclass
class ReliabilityWeights:
    """Weights for reliability scoring."""
    performance: float = 0.30
    calibration: float = 0.25
    stability: float = 0.20
    freshness: float = 0.15
    feature_health: float = 0.10


@dataclass
class AlertConfig:
    """Alert configuration."""
    max_per_hour: int = 100
    deduplicate_window: int = 60


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    refresh_interval: int = 5
    history_length: int = 100
    use_colors: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size: int = 10485760
    backup_count: int = 5


@dataclass
class MetaWatchdogConfig:
    """Complete Meta-Watchdog configuration."""
    name: str = "meta-watchdog"
    version: str = "1.0.0"
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    reliability_weights: ReliabilityWeights = field(default_factory=ReliabilityWeights)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feature_names: List[str] = field(default_factory=list)
    critical_features: List[str] = field(default_factory=list)


class ConfigManager:
    """
    Manages Meta-Watchdog configuration.
    
    Supports:
    - Loading from YAML files
    - Environment variable overrides
    - Programmatic configuration
    """
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config manager.
        
        Args:
            config_path: Path to config file (optional)
        """
        self.config_path = config_path
        self._config: Optional[MetaWatchdogConfig] = None
    
    def load(self) -> MetaWatchdogConfig:
        """Load configuration."""
        # Start with defaults
        self._config = MetaWatchdogConfig()
        
        # Load from file if available
        if self.config_path and os.path.exists(self.config_path):
            self._load_from_file(self.config_path)
        elif os.path.exists(self.DEFAULT_CONFIG_PATH):
            self._load_from_file(str(self.DEFAULT_CONFIG_PATH))
        
        # Apply environment overrides
        self._apply_env_overrides()
        
        return self._config
    
    def _load_from_file(self, path: str) -> None:
        """Load configuration from file."""
        if not YAML_AVAILABLE:
            return
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data:
            self._apply_dict(data)
    
    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """Apply dictionary to configuration."""
        if "system" in data:
            self._config.name = data["system"].get("name", self._config.name)
            self._config.version = data["system"].get("version", self._config.version)
            self._config.logging.level = data["system"].get("log_level", self._config.logging.level)
        
        if "thresholds" in data:
            t = data["thresholds"]
            if "reliability" in t:
                self._config.thresholds.reliability_warning = t["reliability"].get("warning", 60.0)
                self._config.thresholds.reliability_critical = t["reliability"].get("critical", 40.0)
            if "failure_probability" in t:
                self._config.thresholds.failure_alert = t["failure_probability"].get("alert", 0.5)
                self._config.thresholds.failure_critical = t["failure_probability"].get("critical", 0.8)
            if "drift" in t:
                self._config.thresholds.drift_threshold = t["drift"].get("threshold", 0.1)
            if "calibration" in t:
                self._config.thresholds.calibration_error = t["calibration"].get("error_threshold", 0.15)
        
        if "monitoring" in data:
            m = data["monitoring"]
            self._config.monitoring.window_size = m.get("window_size", 100)
            self._config.monitoring.health_check_interval = m.get("health_check_interval", 60)
            self._config.monitoring.deep_analysis_interval = m.get("deep_analysis_interval", 300)
        
        if "reliability_weights" in data:
            w = data["reliability_weights"]
            self._config.reliability_weights.performance = w.get("performance", 0.30)
            self._config.reliability_weights.calibration = w.get("calibration", 0.25)
            self._config.reliability_weights.stability = w.get("stability", 0.20)
            self._config.reliability_weights.freshness = w.get("freshness", 0.15)
            self._config.reliability_weights.feature_health = w.get("feature_health", 0.10)
        
        if "alerts" in data:
            a = data["alerts"]
            self._config.alerts.max_per_hour = a.get("max_per_hour", 100)
            self._config.alerts.deduplicate_window = a.get("deduplicate_window", 60)
        
        if "dashboard" in data:
            d = data["dashboard"]
            self._config.dashboard.refresh_interval = d.get("refresh_interval", 5)
            self._config.dashboard.history_length = d.get("history_length", 100)
            self._config.dashboard.use_colors = d.get("use_colors", True)
        
        if "features" in data:
            f = data["features"]
            self._config.feature_names = f.get("names", [])
            self._config.critical_features = f.get("critical_features", [])
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_map = {
            "MW_RELIABILITY_WARNING": ("thresholds", "reliability_warning", float),
            "MW_RELIABILITY_CRITICAL": ("thresholds", "reliability_critical", float),
            "MW_FAILURE_ALERT": ("thresholds", "failure_alert", float),
            "MW_LOG_LEVEL": ("logging", "level", str),
            "MW_WINDOW_SIZE": ("monitoring", "window_size", int),
        }
        
        for env_var, (section, attr, type_) in env_map.items():
            value = os.environ.get(env_var)
            if value:
                config_section = getattr(self._config, section)
                setattr(config_section, attr, type_(value))
    
    def get(self) -> MetaWatchdogConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load()
        return self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        config = self.get()
        
        return {
            "name": config.name,
            "version": config.version,
            "thresholds": {
                "reliability_warning": config.thresholds.reliability_warning,
                "reliability_critical": config.thresholds.reliability_critical,
                "failure_alert": config.thresholds.failure_alert,
                "failure_critical": config.thresholds.failure_critical,
            },
            "monitoring": {
                "window_size": config.monitoring.window_size,
                "health_check_interval": config.monitoring.health_check_interval,
            },
            "logging": {
                "level": config.logging.level,
            },
        }


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> MetaWatchdogConfig:
    """Get global configuration."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager.get()


def reset_config() -> None:
    """Reset global configuration."""
    global _config_manager
    _config_manager = None
