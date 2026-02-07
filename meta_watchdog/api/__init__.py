"""API module for Meta-Watchdog."""

from meta_watchdog.api.server import (
    APIResponse,
    MetaWatchdogAPIHandler,
    MetaWatchdogAPIServer,
    create_api_server,
)

from meta_watchdog.api.client import (
    ClientError,
    ConnectionError,
    APIError,
    TimeoutError,
    HealthStatus,
    SystemStatus,
    MetricsSummary,
    ReliabilityReport,
    Alert,
    MetaWatchdogClient,
    AsyncMetaWatchdogClient,
    create_client,
)

__all__ = [
    # Server
    "APIResponse",
    "MetaWatchdogAPIHandler",
    "MetaWatchdogAPIServer",
    "create_api_server",
    # Client
    "ClientError",
    "ConnectionError",
    "APIError",
    "TimeoutError",
    "HealthStatus",
    "SystemStatus",
    "MetricsSummary",
    "ReliabilityReport",
    "Alert",
    "MetaWatchdogClient",
    "AsyncMetaWatchdogClient",
    "create_client",
]
