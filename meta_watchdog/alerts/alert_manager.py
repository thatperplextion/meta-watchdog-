"""
Alert Management System for Meta-Watchdog.

This module provides a flexible alerting framework for notifying
stakeholders about model reliability issues, failures, and recommendations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol
import json
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Categories of alerts."""
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    FAILURE_PREDICTION = "failure_prediction"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"
    ACTION_REQUIRED = "action_required"


@dataclass
class Alert:
    """Represents a single alert."""
    
    alert_id: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "meta_watchdog"
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }
    
    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AlertChannel(Protocol):
    """Protocol for alert delivery channels."""
    
    def send(self, alert: Alert) -> bool:
        """Send an alert through this channel."""
        ...
    
    def is_available(self) -> bool:
        """Check if the channel is available."""
        ...


class ConsoleAlertChannel:
    """Sends alerts to console/stdout."""
    
    def __init__(self, colored: bool = True):
        self.colored = colored
        self._colors = {
            AlertSeverity.INFO: "\033[94m",      # Blue
            AlertSeverity.WARNING: "\033[93m",   # Yellow
            AlertSeverity.ERROR: "\033[91m",     # Red
            AlertSeverity.CRITICAL: "\033[95m",  # Magenta
        }
        self._reset = "\033[0m"
    
    def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        try:
            if self.colored:
                color = self._colors.get(alert.severity, "")
                prefix = f"{color}[{alert.severity.value.upper()}]{self._reset}"
            else:
                prefix = f"[{alert.severity.value.upper()}]"
            
            print(f"\n{prefix} {alert.title}")
            print(f"  Category: {alert.category.value}")
            print(f"  Message: {alert.message}")
            print(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if alert.metadata:
                print(f"  Details: {alert.metadata}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False
    
    def is_available(self) -> bool:
        return True


class LogAlertChannel:
    """Sends alerts to logging system."""
    
    def __init__(self, logger_name: str = "meta_watchdog.alerts"):
        self.logger = logging.getLogger(logger_name)
        self._level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }
    
    def send(self, alert: Alert) -> bool:
        """Log the alert."""
        try:
            level = self._level_map.get(alert.severity, logging.INFO)
            self.logger.log(
                level,
                f"[{alert.category.value}] {alert.title}: {alert.message}",
                extra={"alert": alert.to_dict()}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            return False
    
    def is_available(self) -> bool:
        return True


class WebhookAlertChannel:
    """Sends alerts via HTTP webhook."""
    
    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
    
    def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import urllib.request
            
            data = json.dumps(alert.to_dict()).encode("utf-8")
            request = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=self.headers,
                method="POST"
            )
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if webhook is reachable."""
        try:
            import urllib.request
            request = urllib.request.Request(self.webhook_url, method="HEAD")
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status < 400
        except Exception:
            return False


class CallbackAlertChannel:
    """Sends alerts to a callback function."""
    
    def __init__(self, callback: Callable[[Alert], bool]):
        self.callback = callback
    
    def send(self, alert: Alert) -> bool:
        """Execute callback with alert."""
        try:
            return self.callback(alert)
        except Exception as e:
            logger.error(f"Alert callback failed: {e}")
            return False
    
    def is_available(self) -> bool:
        return self.callback is not None


@dataclass
class AlertRule:
    """Rule for automatic alert generation."""
    
    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    category: AlertCategory
    title_template: str
    message_template: str
    cooldown_seconds: int = 60
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate rule and return alert if condition is met."""
        if not self.enabled:
            return None
        
        # Check cooldown
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return None
        
        # Evaluate condition
        try:
            if self.condition(context):
                self.last_triggered = datetime.now()
                return Alert(
                    alert_id=f"{self.rule_id}_{datetime.now().timestamp()}",
                    severity=self.severity,
                    category=self.category,
                    title=self.title_template.format(**context),
                    message=self.message_template.format(**context),
                    metadata={"rule_id": self.rule_id, "context": context}
                )
        except Exception as e:
            logger.error(f"Rule evaluation failed for {self.rule_id}: {e}")
        
        return None


class AlertManager:
    """Central manager for alerts."""
    
    def __init__(self):
        self.channels: Dict[str, AlertChannel] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.max_history: int = 1000
        self._filters: List[Callable[[Alert], bool]] = []
    
    def add_channel(self, name: str, channel: AlertChannel) -> None:
        """Register an alert channel."""
        self.channels[name] = channel
        logger.info(f"Added alert channel: {name}")
    
    def remove_channel(self, name: str) -> None:
        """Remove an alert channel."""
        if name in self.channels:
            del self.channels[name]
            logger.info(f"Removed alert channel: {name}")
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alerting rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove an alerting rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def add_filter(self, filter_func: Callable[[Alert], bool]) -> None:
        """Add a filter to suppress certain alerts."""
        self._filters.append(filter_func)
    
    def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send an alert through specified channels."""
        # Apply filters
        for f in self._filters:
            if not f(alert):
                logger.debug(f"Alert {alert.alert_id} filtered out")
                return {}
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Send through channels
        target_channels = channels or list(self.channels.keys())
        results = {}
        
        for name in target_channels:
            if name in self.channels:
                channel = self.channels[name]
                if channel.is_available():
                    results[name] = channel.send(alert)
                else:
                    results[name] = False
                    logger.warning(f"Channel {name} is not available")
        
        return results
    
    def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules and send generated alerts."""
        alerts = []
        
        for rule in self.rules.values():
            alert = rule.evaluate(context)
            if alert:
                self.send_alert(alert)
                alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
    ) -> List[Alert]:
        """Get unresolved alerts, optionally filtered."""
        alerts = [a for a in self.alert_history if not a.resolved]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        return alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total = len(self.alert_history)
        active = len([a for a in self.alert_history if not a.resolved])
        
        by_severity = {}
        by_category = {}
        
        for alert in self.alert_history:
            sev = alert.severity.value
            cat = alert.category.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1
        
        return {
            "total_alerts": total,
            "active_alerts": active,
            "resolved_alerts": total - active,
            "by_severity": by_severity,
            "by_category": by_category,
            "channels": list(self.channels.keys()),
            "rules": list(self.rules.keys()),
        }


# Default alert rules
def create_default_rules() -> List[AlertRule]:
    """Create default alerting rules."""
    return [
        AlertRule(
            rule_id="reliability_warning",
            name="Reliability Warning",
            condition=lambda ctx: ctx.get("reliability_score", 100) < 70,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RELIABILITY,
            title_template="Model Reliability Degraded",
            message_template="Reliability score dropped to {reliability_score:.1f}/100",
            cooldown_seconds=300,
        ),
        AlertRule(
            rule_id="reliability_critical",
            name="Reliability Critical",
            condition=lambda ctx: ctx.get("reliability_score", 100) < 50,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RELIABILITY,
            title_template="Critical Reliability Alert",
            message_template="Reliability score is critically low at {reliability_score:.1f}/100",
            cooldown_seconds=60,
        ),
        AlertRule(
            rule_id="failure_imminent",
            name="Failure Imminent",
            condition=lambda ctx: ctx.get("failure_probability", 0) > 0.7,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.FAILURE_PREDICTION,
            title_template="Model Failure Imminent",
            message_template="Failure probability is {failure_probability:.1%}. Immediate action required.",
            cooldown_seconds=60,
        ),
        AlertRule(
            rule_id="performance_degradation",
            name="Performance Degradation",
            condition=lambda ctx: ctx.get("accuracy", 1.0) < 0.8,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PERFORMANCE,
            title_template="Performance Degradation Detected",
            message_template="Model accuracy dropped to {accuracy:.1%}",
            cooldown_seconds=300,
        ),
    ]
