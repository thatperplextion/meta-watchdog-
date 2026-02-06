"""
Logging and Telemetry Module

Provides structured logging and metrics collection for Meta-Watchdog.
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class LogLevel(Enum):
    """Log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None


@dataclass
class MetricEvent:
    """Telemetry metric event."""
    timestamp: datetime
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class StructuredLogger:
    """
    Structured logger for Meta-Watchdog.
    
    Provides:
    - Structured log events with context
    - JSON formatting for log aggregation
    - Component-specific logging
    - Correlation IDs for tracing
    """
    
    def __init__(
        self,
        name: str = "meta_watchdog",
        level: LogLevel = LogLevel.INFO,
        json_format: bool = False,
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Minimum log level
            json_format: Whether to output JSON
        """
        self.name = name
        self.level = level
        self.json_format = json_format
        
        # Setup Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
        
        # Setup handler if not exists
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            
            if json_format:
                formatter = logging.Formatter('%(message)s')
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        
        # Event history
        self._events: List[LogEvent] = []
        self._max_history = 1000
        
        # Correlation ID
        self._correlation_id: Optional[str] = None
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for tracing."""
        self._correlation_id = correlation_id
    
    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        self._correlation_id = None
    
    def _log(
        self,
        level: LogLevel,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal log method."""
        event = LogEvent(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            data=data,
            correlation_id=self._correlation_id,
        )
        
        # Store event
        self._events.append(event)
        if len(self._events) > self._max_history:
            self._events = self._events[-self._max_history:]
        
        # Format message
        if self.json_format:
            log_dict = {
                "timestamp": event.timestamp.isoformat(),
                "level": level.name,
                "component": component,
                "message": message,
            }
            if data:
                log_dict["data"] = data
            if self._correlation_id:
                log_dict["correlation_id"] = self._correlation_id
            
            formatted = json.dumps(log_dict)
        else:
            formatted = f"[{component}] {message}"
            if data:
                formatted += f" | {data}"
        
        # Log to Python logger
        self._logger.log(level.value, formatted)
    
    def debug(self, component: str, message: str, data: Optional[Dict] = None) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, component, message, data)
    
    def info(self, component: str, message: str, data: Optional[Dict] = None) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, component, message, data)
    
    def warning(self, component: str, message: str, data: Optional[Dict] = None) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, component, message, data)
    
    def error(self, component: str, message: str, data: Optional[Dict] = None) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, component, message, data)
    
    def critical(self, component: str, message: str, data: Optional[Dict] = None) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, component, message, data)
    
    def get_events(
        self,
        level: Optional[LogLevel] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[LogEvent]:
        """Get log events with optional filtering."""
        events = self._events
        
        if level:
            events = [e for e in events if e.level == level]
        
        if component:
            events = [e for e in events if e.component == component]
        
        return events[-limit:]


class TelemetryCollector:
    """
    Collects telemetry metrics for Meta-Watchdog.
    
    Tracks:
    - Observation counts
    - Reliability scores over time
    - Alert frequencies
    - Component latencies
    """
    
    def __init__(self):
        """Initialize the telemetry collector."""
        self._metrics: Dict[str, List[MetricEvent]] = {}
        self._max_history = 10000
        self._callbacks: List[Callable[[MetricEvent], None]] = []
    
    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
    ) -> None:
        """Record a metric."""
        event = MetricEvent(
            timestamp=datetime.now(),
            name=name,
            value=value,
            tags=tags or {},
            unit=unit,
        )
        
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append(event)
        
        # Trim history
        if len(self._metrics[name]) > self._max_history:
            self._metrics[name] = self._metrics[name][-self._max_history:]
        
        # Invoke callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        self.record(name, value, tags, unit="count")
    
    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a gauge metric."""
        self.record(name, value, tags, unit="gauge")
    
    def timing(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timing metric in milliseconds."""
        self.record(name, value, tags, unit="ms")
    
    def get_metrics(
        self,
        name: str,
        limit: int = 100,
    ) -> List[MetricEvent]:
        """Get metrics by name."""
        return self._metrics.get(name, [])[-limit:]
    
    def get_latest(self, name: str) -> Optional[MetricEvent]:
        """Get latest value for a metric."""
        events = self._metrics.get(name, [])
        return events[-1] if events else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for name, events in self._metrics.items():
            if events:
                values = [e.value for e in events[-100:]]
                summary[name] = {
                    "latest": events[-1].value,
                    "count": len(events),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        
        return summary
    
    def register_callback(self, callback: Callable[[MetricEvent], None]) -> None:
        """Register callback for new metrics."""
        self._callbacks.append(callback)
    
    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()


class MetaWatchdogTelemetry:
    """
    Combined logging and telemetry for Meta-Watchdog.
    
    Usage:
        telemetry = MetaWatchdogTelemetry()
        
        telemetry.log.info("monitor", "Observation processed", {"count": 100})
        telemetry.metrics.gauge("reliability_score", 85.5)
    """
    
    def __init__(
        self,
        log_level: LogLevel = LogLevel.INFO,
        json_logs: bool = False,
    ):
        """
        Initialize telemetry system.
        
        Args:
            log_level: Minimum log level
            json_logs: Whether to output JSON logs
        """
        self.log = StructuredLogger(
            name="meta_watchdog",
            level=log_level,
            json_format=json_logs,
        )
        self.metrics = TelemetryCollector()
    
    # Convenience methods for common operations
    
    def observation_processed(
        self,
        count: int = 1,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log observation processing."""
        self.metrics.increment("observations_total", count)
        
        if latency_ms:
            self.metrics.timing("observation_latency", latency_ms)
        
        self.log.debug("monitor", f"Processed {count} observations")
    
    def reliability_updated(
        self,
        score: float,
        components: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log reliability score update."""
        self.metrics.gauge("reliability_score", score)
        
        if components:
            for name, value in components.items():
                self.metrics.gauge(f"reliability_{name}", value)
        
        self.log.info("reliability", f"Reliability score: {score:.1f}", components)
    
    def alert_raised(
        self,
        level: str,
        component: str,
        message: str,
    ) -> None:
        """Log alert."""
        self.metrics.increment("alerts_total", tags={"level": level})
        
        self.log.warning("alerts", f"Alert [{level}] from {component}: {message}")
    
    def analysis_completed(
        self,
        analysis_type: str,
        latency_ms: float,
        findings: int = 0,
    ) -> None:
        """Log analysis completion."""
        self.metrics.timing(f"analysis_{analysis_type}_latency", latency_ms)
        self.metrics.increment(f"analysis_{analysis_type}_total")
        
        self.log.info(
            "analysis",
            f"{analysis_type} analysis completed",
            {"latency_ms": latency_ms, "findings": findings}
        )


# Global telemetry instance
_telemetry: Optional[MetaWatchdogTelemetry] = None


def get_telemetry() -> MetaWatchdogTelemetry:
    """Get global telemetry instance."""
    global _telemetry
    
    if _telemetry is None:
        _telemetry = MetaWatchdogTelemetry()
    
    return _telemetry


def reset_telemetry() -> None:
    """Reset global telemetry."""
    global _telemetry
    _telemetry = None
