"""Alert management module for Meta-Watchdog."""

from meta_watchdog.alerts.alert_manager import (
    Alert,
    AlertCategory,
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
    CallbackAlertChannel,
    ConsoleAlertChannel,
    LogAlertChannel,
    WebhookAlertChannel,
    create_default_rules,
)

__all__ = [
    "Alert",
    "AlertCategory",
    "AlertChannel",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "CallbackAlertChannel",
    "ConsoleAlertChannel",
    "LogAlertChannel",
    "WebhookAlertChannel",
    "create_default_rules",
]
