"""
Terminal Dashboard

Rich terminal-based dashboard for Meta-Watchdog using ASCII/ANSI.
Works in any terminal without external dependencies.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from meta_watchdog.dashboard.data_provider import DashboardDataProvider


class TerminalDashboard:
    """
    ASCII-based terminal dashboard for Meta-Watchdog.
    
    Displays:
    - System health status
    - Reliability metrics
    - Active alerts
    - Top recommendations
    """
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    
    def __init__(
        self,
        data_provider: DashboardDataProvider,
        use_colors: bool = True,
        width: int = 80,
    ):
        """
        Initialize the terminal dashboard.
        
        Args:
            data_provider: Dashboard data provider
            use_colors: Whether to use ANSI colors
            width: Terminal width
        """
        self.data_provider = data_provider
        self.use_colors = use_colors
        self.width = width
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _status_color(self, status: str) -> str:
        """Get color for status."""
        if status in ("healthy", "excellent", "good", "low"):
            return "green"
        elif status in ("warning", "fair", "moderate", "degraded"):
            return "yellow"
        else:
            return "red"
    
    def _bar(self, value: float, max_val: float = 100, length: int = 20) -> str:
        """Create a progress bar."""
        filled = int((value / max_val) * length)
        empty = length - filled
        
        if value >= 70:
            color = "green"
        elif value >= 40:
            color = "yellow"
        else:
            color = "red"
        
        bar = "█" * filled + "░" * empty
        return self._color(bar, color)
    
    def _box(self, title: str, content: List[str]) -> str:
        """Create a box with content."""
        inner_width = self.width - 4
        
        lines = [
            "┌" + "─" * (inner_width + 2) + "┐",
            "│ " + self._color(title.center(inner_width), "bold") + " │",
            "├" + "─" * (inner_width + 2) + "┤",
        ]
        
        for line in content:
            # Truncate if too long (accounting for color codes)
            visible_len = len(line.replace("\033[0m", "").replace("\033[1m", "")
                             .replace("\033[91m", "").replace("\033[92m", "")
                             .replace("\033[93m", "").replace("\033[94m", "")
                             .replace("\033[96m", "").replace("\033[97m", ""))
            
            if visible_len > inner_width:
                # Simple truncation
                line = line[:inner_width-3] + "..."
            
            padding = inner_width - visible_len
            lines.append(f"│ {line}{' ' * max(0, padding)} │")
        
        lines.append("└" + "─" * (inner_width + 2) + "┘")
        
        return "\n".join(lines)
    
    def render(self) -> str:
        """Render the complete dashboard."""
        data = self.data_provider.get_dashboard_data()
        
        sections = [
            self._render_header(),
            self._render_status(data),
            self._render_metrics(data),
            self._render_alerts(data),
            self._render_recommendations(data),
            self._render_footer(),
        ]
        
        return "\n\n".join(sections)
    
    def _render_header(self) -> str:
        """Render dashboard header."""
        title = "META-WATCHDOG DASHBOARD"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = "═" * self.width
        title_line = self._color(title.center(self.width), "bold")
        time_line = timestamp.center(self.width)
        
        return f"{header}\n{title_line}\n{time_line}\n{header}"
    
    def _render_status(self, data) -> str:
        """Render system status."""
        status_color = self._status_color(data.system_status)
        
        content = [
            f"Status: {self._color(data.system_status.upper(), status_color)}",
            "",
            f"Reliability: {self._bar(data.reliability_score)} {data.reliability_score:.1f}/100",
            f"Failure Risk: {self._bar(data.failure_risk * 100)} {data.failure_risk:.1%}",
        ]
        
        return self._box("SYSTEM STATUS", content)
    
    def _render_metrics(self, data) -> str:
        """Render metrics panel."""
        content = []
        
        for trend in data.metric_trends:
            arrow = "↑" if trend.trend_direction == "up" else "↓" if trend.trend_direction == "down" else "→"
            color = self._status_color("good" if trend.trend_direction != "down" else "warning")
            
            value_str = f"{trend.current_value:.2f}" if isinstance(trend.current_value, float) else str(trend.current_value)
            change_str = f"{trend.change_percent:+.1f}%"
            
            content.append(
                f"{trend.name:15} {value_str:8} {self._color(arrow + ' ' + change_str, color)}"
            )
        
        return self._box("METRICS", content) if content else ""
    
    def _render_alerts(self, data) -> str:
        """Render alerts panel."""
        if not data.active_alerts:
            return self._box("ALERTS", [self._color("No active alerts", "green")])
        
        content = [
            f"Critical: {self._color(str(data.alert_summary['critical']), 'red')}  "
            f"Warning: {self._color(str(data.alert_summary['warning']), 'yellow')}  "
            f"Info: {data.alert_summary['info']}",
            "",
        ]
        
        for alert in data.active_alerts[:5]:
            level_color = "red" if alert["level"] == "critical" else "yellow" if alert["level"] == "warning" else "blue"
            content.append(
                f"[{self._color(alert['level'].upper(), level_color)}] {alert['message'][:50]}"
            )
        
        return self._box("ALERTS", content)
    
    def _render_recommendations(self, data) -> str:
        """Render recommendations panel."""
        if not data.top_recommendations:
            return self._box("RECOMMENDATIONS", ["No recommendations at this time"])
        
        content = []
        for i, rec in enumerate(data.top_recommendations[:3], 1):
            priority_color = "red" if rec["priority"] == "critical" else "yellow" if rec["priority"] == "high" else "blue"
            content.append(
                f"{i}. [{self._color(rec['priority'].upper(), priority_color)}] {rec['title'][:45]}"
            )
            content.append(f"   Impact: {rec['impact']:.0%}")
        
        return self._box("TOP RECOMMENDATIONS", content)
    
    def _render_footer(self) -> str:
        """Render dashboard footer."""
        return "─" * self.width + "\n" + "Press 'q' to quit | 'r' to refresh | 'd' for deep analysis".center(self.width)
    
    def render_compact(self) -> str:
        """Render a compact single-line status."""
        summary = self.data_provider.get_health_summary()
        
        status_color = self._status_color(summary["status"])
        risk_color = self._status_color(summary["risk_label"].lower())
        
        return (
            f"[Meta-Watchdog] "
            f"Status: {self._color(summary['score_label'], status_color)} ({summary['score']:.0f}) | "
            f"Risk: {self._color(summary['risk_label'], risk_color)} ({summary['risk']:.0%}) | "
            f"Alerts: {summary['alerts']}"
        )
