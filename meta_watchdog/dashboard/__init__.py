"""
Dashboard module for Meta-Watchdog visualization.

This module provides tools for visualizing system health,
alerts, and recommendations.
"""

from meta_watchdog.dashboard.data_provider import (
    DashboardDataProvider,
    DashboardData,
    MetricTrend,
    ComponentStatus,
)
from meta_watchdog.dashboard.terminal import TerminalDashboard

__all__ = [
    "DashboardDataProvider",
    "DashboardData",
    "MetricTrend",
    "ComponentStatus",
    "TerminalDashboard",
]
