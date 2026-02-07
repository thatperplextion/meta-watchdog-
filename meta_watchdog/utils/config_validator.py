"""
Configuration Validation Module for Meta-Watchdog

Provides configuration schema validation, defaults, and environment variable support.
"""

from typing import Dict, List, Optional, Any, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import re
import json


class ValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, message: str, path: str = "", errors: Optional[List[str]] = None):
        super().__init__(message)
        self.path = path
        self.errors = errors or [message]


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


@dataclass
class ConfigField:
    """Configuration field definition."""
    name: str
    type: ConfigType
    required: bool = False
    default: Any = None
    description: str = ""
    env_var: Optional[str] = None
    validator: Optional[Callable[[Any], bool]] = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    
    def validate(self, value: Any) -> List[str]:
        """Validate a value against this field definition."""
        errors = []
        
        if value is None:
            if self.required:
                errors.append(f"Field '{self.name}' is required")
            return errors
        
        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            errors.append(type_error)
            return errors
        
        # Range validation for numbers
        if self.type in (ConfigType.INTEGER, ConfigType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"Field '{self.name}' must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"Field '{self.name}' must be <= {self.max_value}")
        
        # Choices validation
        if self.choices is not None and value not in self.choices:
            errors.append(f"Field '{self.name}' must be one of {self.choices}")
        
        # Pattern validation for strings
        if self.type == ConfigType.STRING and self.pattern:
            if not re.match(self.pattern, str(value)):
                errors.append(f"Field '{self.name}' does not match pattern '{self.pattern}'")
        
        # Custom validator
        if self.validator:
            try:
                if not self.validator(value):
                    errors.append(f"Field '{self.name}' failed custom validation")
            except Exception as e:
                errors.append(f"Field '{self.name}' validation error: {str(e)}")
        
        return errors
    
    def _validate_type(self, value: Any) -> tuple:
        """Validate the type of a value."""
        if self.type == ConfigType.ANY:
            return True, ""
        
        type_map = {
            ConfigType.STRING: str,
            ConfigType.INTEGER: int,
            ConfigType.FLOAT: (int, float),
            ConfigType.BOOLEAN: bool,
            ConfigType.LIST: list,
            ConfigType.DICT: dict,
        }
        
        expected = type_map.get(self.type)
        if expected and not isinstance(value, expected):
            return False, f"Field '{self.name}' must be of type {self.type.value}"
        
        return True, ""
    
    def coerce(self, value: Any) -> Any:
        """Coerce a value to the expected type."""
        if value is None:
            return self.default
        
        try:
            if self.type == ConfigType.STRING:
                return str(value)
            elif self.type == ConfigType.INTEGER:
                return int(value)
            elif self.type == ConfigType.FLOAT:
                return float(value)
            elif self.type == ConfigType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif self.type == ConfigType.LIST:
                if isinstance(value, str):
                    return json.loads(value)
                return list(value)
            elif self.type == ConfigType.DICT:
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
            else:
                return value
        except (ValueError, TypeError, json.JSONDecodeError):
            return value


@dataclass
class ConfigSection:
    """Configuration section with multiple fields."""
    name: str
    fields: List[ConfigField] = field(default_factory=list)
    description: str = ""
    
    def add_field(self, field_def: ConfigField):
        """Add a field to this section."""
        self.fields.append(field_def)
    
    def get_field(self, name: str) -> Optional[ConfigField]:
        """Get a field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration dictionary against this section."""
        errors = []
        for field_def in self.fields:
            value = config.get(field_def.name)
            field_errors = field_def.validate(value)
            errors.extend(field_errors)
        return errors


class ConfigSchema:
    """Configuration schema definition."""
    
    def __init__(self, name: str = "config"):
        self.name = name
        self.sections: Dict[str, ConfigSection] = {}
        self.root_fields: List[ConfigField] = []
    
    def add_section(self, name: str, description: str = "") -> ConfigSection:
        """Add a configuration section."""
        section = ConfigSection(name=name, description=description)
        self.sections[name] = section
        return section
    
    def add_field(
        self,
        name: str,
        type: ConfigType = ConfigType.STRING,
        required: bool = False,
        default: Any = None,
        description: str = "",
        env_var: Optional[str] = None,
        section: Optional[str] = None,
        **kwargs
    ) -> ConfigField:
        """Add a configuration field."""
        field_def = ConfigField(
            name=name,
            type=type,
            required=required,
            default=default,
            description=description,
            env_var=env_var,
            **kwargs
        )
        
        if section:
            if section not in self.sections:
                self.add_section(section)
            self.sections[section].add_field(field_def)
        else:
            self.root_fields.append(field_def)
        
        return field_def
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration dictionary."""
        errors = []
        
        # Validate root fields
        for field_def in self.root_fields:
            value = config.get(field_def.name)
            errors.extend(field_def.validate(value))
        
        # Validate sections
        for section_name, section in self.sections.items():
            section_config = config.get(section_name, {})
            if not isinstance(section_config, dict):
                errors.append(f"Section '{section_name}' must be a dictionary")
                continue
            errors.extend(section.validate(section_config))
        
        return errors
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        defaults = {}
        
        for field_def in self.root_fields:
            if field_def.default is not None:
                defaults[field_def.name] = field_def.default
        
        for section_name, section in self.sections.items():
            section_defaults = {}
            for field_def in section.fields:
                if field_def.default is not None:
                    section_defaults[field_def.name] = field_def.default
            if section_defaults:
                defaults[section_name] = section_defaults
        
        return defaults


class ConfigValidator:
    """Configuration validator with environment variable support."""
    
    def __init__(self, schema: ConfigSchema):
        self.schema = schema
    
    def load_from_env(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        result = config.copy() if config else {}
        
        # Load root fields
        for field_def in self.schema.root_fields:
            if field_def.env_var:
                env_value = os.environ.get(field_def.env_var)
                if env_value is not None:
                    result[field_def.name] = field_def.coerce(env_value)
        
        # Load section fields
        for section_name, section in self.schema.sections.items():
            if section_name not in result:
                result[section_name] = {}
            for field_def in section.fields:
                if field_def.env_var:
                    env_value = os.environ.get(field_def.env_var)
                    if env_value is not None:
                        result[section_name][field_def.name] = field_def.coerce(env_value)
        
        return result
    
    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration."""
        result = self.schema.get_defaults()
        
        # Deep merge with provided config
        def deep_merge(base: Dict, override: Dict) -> Dict:
            merged = base.copy()
            for key, value in override.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged
        
        return deep_merge(result, config)
    
    def validate(self, config: Dict[str, Any], raise_on_error: bool = True) -> List[str]:
        """Validate configuration."""
        errors = self.schema.validate(config)
        
        if errors and raise_on_error:
            raise ValidationError(
                f"Configuration validation failed with {len(errors)} errors",
                errors=errors
            )
        
        return errors
    
    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration: apply defaults, load env vars, validate."""
        result = self.apply_defaults(config)
        result = self.load_from_env(result)
        self.validate(result)
        return result


# Pre-defined Meta-Watchdog configuration schema
def create_meta_watchdog_schema() -> ConfigSchema:
    """Create the default Meta-Watchdog configuration schema."""
    schema = ConfigSchema("meta_watchdog")
    
    # General settings
    schema.add_field(
        "debug",
        type=ConfigType.BOOLEAN,
        default=False,
        description="Enable debug mode",
        env_var="META_WATCHDOG_DEBUG"
    )
    
    schema.add_field(
        "log_level",
        type=ConfigType.STRING,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        description="Logging level",
        env_var="META_WATCHDOG_LOG_LEVEL"
    )
    
    # Monitoring section
    monitoring = schema.add_section("monitoring", "Monitoring configuration")
    monitoring.add_field(ConfigField(
        name="enabled",
        type=ConfigType.BOOLEAN,
        default=True,
        description="Enable monitoring"
    ))
    monitoring.add_field(ConfigField(
        name="window_size",
        type=ConfigType.INTEGER,
        default=100,
        min_value=10,
        max_value=10000,
        description="Size of the monitoring window",
        env_var="META_WATCHDOG_WINDOW_SIZE"
    ))
    monitoring.add_field(ConfigField(
        name="check_interval",
        type=ConfigType.FLOAT,
        default=60.0,
        min_value=1.0,
        description="Health check interval in seconds"
    ))
    
    # Drift detection section
    drift = schema.add_section("drift", "Drift detection configuration")
    drift.add_field(ConfigField(
        name="enabled",
        type=ConfigType.BOOLEAN,
        default=True,
        description="Enable drift detection"
    ))
    drift.add_field(ConfigField(
        name="threshold",
        type=ConfigType.FLOAT,
        default=0.05,
        min_value=0.0,
        max_value=1.0,
        description="Drift detection threshold",
        env_var="META_WATCHDOG_DRIFT_THRESHOLD"
    ))
    drift.add_field(ConfigField(
        name="detection_method",
        type=ConfigType.STRING,
        default="ks_test",
        choices=["ks_test", "psi", "chi_square", "wasserstein"],
        description="Drift detection method"
    ))
    
    # Reliability section
    reliability = schema.add_section("reliability", "Reliability scoring configuration")
    reliability.add_field(ConfigField(
        name="min_samples",
        type=ConfigType.INTEGER,
        default=30,
        min_value=1,
        description="Minimum samples for reliability calculation"
    ))
    reliability.add_field(ConfigField(
        name="decay_factor",
        type=ConfigType.FLOAT,
        default=0.95,
        min_value=0.0,
        max_value=1.0,
        description="Exponential decay factor for historical scores"
    ))
    
    # Alerts section
    alerts = schema.add_section("alerts", "Alerting configuration")
    alerts.add_field(ConfigField(
        name="enabled",
        type=ConfigType.BOOLEAN,
        default=True,
        description="Enable alerting"
    ))
    alerts.add_field(ConfigField(
        name="cooldown_seconds",
        type=ConfigType.INTEGER,
        default=300,
        min_value=0,
        description="Alert cooldown period"
    ))
    alerts.add_field(ConfigField(
        name="max_queue_size",
        type=ConfigType.INTEGER,
        default=1000,
        min_value=10,
        description="Maximum alert queue size"
    ))
    
    # API section
    api = schema.add_section("api", "API server configuration")
    api.add_field(ConfigField(
        name="enabled",
        type=ConfigType.BOOLEAN,
        default=False,
        description="Enable API server"
    ))
    api.add_field(ConfigField(
        name="host",
        type=ConfigType.STRING,
        default="0.0.0.0",
        description="API server host",
        env_var="META_WATCHDOG_API_HOST"
    ))
    api.add_field(ConfigField(
        name="port",
        type=ConfigType.INTEGER,
        default=8080,
        min_value=1,
        max_value=65535,
        description="API server port",
        env_var="META_WATCHDOG_API_PORT"
    ))
    
    # Persistence section
    persistence = schema.add_section("persistence", "State persistence configuration")
    persistence.add_field(ConfigField(
        name="enabled",
        type=ConfigType.BOOLEAN,
        default=False,
        description="Enable state persistence"
    ))
    persistence.add_field(ConfigField(
        name="checkpoint_dir",
        type=ConfigType.STRING,
        default="./checkpoints",
        description="Checkpoint directory path",
        env_var="META_WATCHDOG_CHECKPOINT_DIR"
    ))
    persistence.add_field(ConfigField(
        name="auto_checkpoint",
        type=ConfigType.BOOLEAN,
        default=True,
        description="Enable automatic checkpointing"
    ))
    persistence.add_field(ConfigField(
        name="checkpoint_interval",
        type=ConfigType.INTEGER,
        default=3600,
        min_value=60,
        description="Auto-checkpoint interval in seconds"
    ))
    
    return schema


# Global default schema
default_schema = create_meta_watchdog_schema()


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration using default schema."""
    validator = ConfigValidator(default_schema)
    return validator.validate(config, raise_on_error=False)


def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process configuration using default schema."""
    validator = ConfigValidator(default_schema)
    return validator.process(config)


__all__ = [
    "ValidationError",
    "ConfigType",
    "ConfigField",
    "ConfigSection",
    "ConfigSchema",
    "ConfigValidator",
    "create_meta_watchdog_schema",
    "default_schema",
    "validate_config",
    "process_config",
]
