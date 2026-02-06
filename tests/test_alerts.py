"""
Tests for the alerts module.
"""

import pytest
from datetime import datetime, timedelta

from meta_watchdog.alerts import (
    Alert,
    AlertCategory,
    AlertManager,
    AlertRule,
    AlertSeverity,
    ConsoleAlertChannel,
    LogAlertChannel,
    CallbackAlertChannel,
    create_default_rules,
)


class TestAlert:
    """Tests for Alert class."""
    
    def test_alert_creation(self):
        """Test basic alert creation."""
        alert = Alert(
            alert_id="test_001",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RELIABILITY,
            title="Test Alert",
            message="This is a test alert",
        )
        
        assert alert.alert_id == "test_001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.category == AlertCategory.RELIABILITY
        assert not alert.resolved
    
    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            alert_id="test_002",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.FAILURE_PREDICTION,
            title="Critical Alert",
            message="Critical issue detected",
            metadata={"score": 45.0},
        )
        
        data = alert.to_dict()
        
        assert data["alert_id"] == "test_002"
        assert data["severity"] == "critical"
        assert data["category"] == "failure_prediction"
        assert data["metadata"]["score"] == 45.0
    
    def test_alert_to_json(self):
        """Test alert JSON serialization."""
        alert = Alert(
            alert_id="test_003",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title="Info Alert",
            message="System information",
        )
        
        json_str = alert.to_json()
        
        assert "test_003" in json_str
        assert "info" in json_str


class TestAlertChannels:
    """Tests for alert channels."""
    
    def test_console_channel(self, capsys):
        """Test console alert channel."""
        channel = ConsoleAlertChannel(colored=False)
        alert = Alert(
            alert_id="console_test",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PERFORMANCE,
            title="Console Test",
            message="Testing console output",
        )
        
        assert channel.is_available()
        result = channel.send(alert)
        
        assert result is True
        captured = capsys.readouterr()
        assert "Console Test" in captured.out
    
    def test_log_channel(self, caplog):
        """Test log alert channel."""
        channel = LogAlertChannel()
        alert = Alert(
            alert_id="log_test",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.DATA_QUALITY,
            title="Log Test",
            message="Testing log output",
        )
        
        assert channel.is_available()
        result = channel.send(alert)
        
        assert result is True
    
    def test_callback_channel(self):
        """Test callback alert channel."""
        received_alerts = []
        
        def callback(alert):
            received_alerts.append(alert)
            return True
        
        channel = CallbackAlertChannel(callback)
        alert = Alert(
            alert_id="callback_test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title="Callback Test",
            message="Testing callback",
        )
        
        result = channel.send(alert)
        
        assert result is True
        assert len(received_alerts) == 1
        assert received_alerts[0].alert_id == "callback_test"


class TestAlertRule:
    """Tests for alert rules."""
    
    def test_rule_evaluation_true(self):
        """Test rule evaluation when condition is true."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            condition=lambda ctx: ctx.get("score", 100) < 70,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RELIABILITY,
            title_template="Score Alert",
            message_template="Score is {score}",
            cooldown_seconds=0,
        )
        
        alert = rule.evaluate({"score": 50})
        
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
    
    def test_rule_evaluation_false(self):
        """Test rule evaluation when condition is false."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            condition=lambda ctx: ctx.get("score", 100) < 70,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RELIABILITY,
            title_template="Score Alert",
            message_template="Score is {score}",
        )
        
        alert = rule.evaluate({"score": 80})
        
        assert alert is None
    
    def test_rule_cooldown(self):
        """Test rule cooldown period."""
        rule = AlertRule(
            rule_id="cooldown_rule",
            name="Cooldown Rule",
            condition=lambda ctx: True,
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title_template="Cooldown Test",
            message_template="Testing cooldown",
            cooldown_seconds=60,
        )
        
        # First evaluation should trigger
        alert1 = rule.evaluate({})
        assert alert1 is not None
        
        # Second evaluation within cooldown should not trigger
        alert2 = rule.evaluate({})
        assert alert2 is None
    
    def test_disabled_rule(self):
        """Test disabled rule."""
        rule = AlertRule(
            rule_id="disabled_rule",
            name="Disabled Rule",
            condition=lambda ctx: True,
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title_template="Disabled Test",
            message_template="This should not trigger",
            enabled=False,
        )
        
        alert = rule.evaluate({})
        
        assert alert is None


class TestAlertManager:
    """Tests for AlertManager."""
    
    @pytest.fixture
    def manager(self):
        """Create alert manager for testing."""
        return AlertManager()
    
    def test_add_channel(self, manager):
        """Test adding alert channel."""
        channel = ConsoleAlertChannel(colored=False)
        manager.add_channel("console", channel)
        
        assert "console" in manager.channels
    
    def test_add_rule(self, manager):
        """Test adding alert rule."""
        rule = AlertRule(
            rule_id="test",
            name="Test",
            condition=lambda ctx: True,
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title_template="Test",
            message_template="Test",
        )
        
        manager.add_rule(rule)
        
        assert "test" in manager.rules
    
    def test_send_alert(self, manager):
        """Test sending alerts."""
        received = []
        channel = CallbackAlertChannel(lambda a: received.append(a) or True)
        manager.add_channel("callback", channel)
        
        alert = Alert(
            alert_id="send_test",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RELIABILITY,
            title="Send Test",
            message="Testing send",
        )
        
        results = manager.send_alert(alert)
        
        assert results["callback"] is True
        assert len(received) == 1
        assert len(manager.alert_history) == 1
    
    def test_evaluate_rules(self, manager):
        """Test rule evaluation."""
        received = []
        channel = CallbackAlertChannel(lambda a: received.append(a) or True)
        manager.add_channel("callback", channel)
        
        rule = AlertRule(
            rule_id="eval_test",
            name="Evaluation Test",
            condition=lambda ctx: ctx.get("reliability", 100) < 70,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RELIABILITY,
            title_template="Low Reliability",
            message_template="Score: {reliability}",
            cooldown_seconds=0,
        )
        manager.add_rule(rule)
        
        alerts = manager.evaluate_rules({"reliability": 50})
        
        assert len(alerts) == 1
        assert len(received) == 1
    
    def test_resolve_alert(self, manager):
        """Test alert resolution."""
        alert = Alert(
            alert_id="resolve_test",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RELIABILITY,
            title="Resolution Test",
            message="Testing resolution",
        )
        
        manager.send_alert(alert, channels=[])
        
        assert not alert.resolved
        
        result = manager.resolve_alert("resolve_test")
        
        assert result is True
        assert manager.alert_history[0].resolved
    
    def test_get_active_alerts(self, manager):
        """Test getting active alerts."""
        for i in range(3):
            alert = Alert(
                alert_id=f"active_test_{i}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.RELIABILITY,
                title=f"Active Test {i}",
                message="Testing active alerts",
            )
            manager.send_alert(alert, channels=[])
        
        # Resolve one
        manager.resolve_alert("active_test_1")
        
        active = manager.get_active_alerts()
        
        assert len(active) == 2
    
    def test_alert_filter(self, manager):
        """Test alert filtering."""
        # Filter out INFO severity
        manager.add_filter(lambda a: a.severity != AlertSeverity.INFO)
        
        info_alert = Alert(
            alert_id="filtered",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title="Filtered",
            message="Should be filtered",
        )
        
        results = manager.send_alert(info_alert)
        
        assert len(results) == 0
        assert len(manager.alert_history) == 0
    
    def test_statistics(self, manager):
        """Test alert statistics."""
        for severity in [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.ERROR]:
            alert = Alert(
                alert_id=f"stats_{severity.value}",
                severity=severity,
                category=AlertCategory.SYSTEM,
                title=f"Stats {severity.value}",
                message="Testing statistics",
            )
            manager.send_alert(alert, channels=[])
        
        stats = manager.get_statistics()
        
        assert stats["total_alerts"] == 3
        assert "info" in stats["by_severity"]
        assert "warning" in stats["by_severity"]
        assert "error" in stats["by_severity"]


class TestDefaultRules:
    """Tests for default alert rules."""
    
    def test_create_default_rules(self):
        """Test default rules creation."""
        rules = create_default_rules()
        
        assert len(rules) > 0
        assert any(r.rule_id == "reliability_warning" for r in rules)
        assert any(r.rule_id == "reliability_critical" for r in rules)
        assert any(r.rule_id == "failure_imminent" for r in rules)
