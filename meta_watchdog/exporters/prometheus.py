"""
Prometheus Metrics Exporter for Meta-Watchdog

Exports system metrics in Prometheus format for monitoring and alerting.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import time
import http.server
import socketserver


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None


@dataclass
class Metric:
    """A Prometheus metric definition."""
    name: str
    type: MetricType
    help: str
    values: List[MetricValue] = field(default_factory=list)
    
    def to_prometheus(self) -> str:
        """Convert metric to Prometheus exposition format."""
        lines = []
        lines.append(f"# HELP {self.name} {self.help}")
        lines.append(f"# TYPE {self.name} {self.type.value}")
        
        for mv in self.values:
            label_str = ""
            if mv.labels:
                label_pairs = [f'{k}="{v}"' for k, v in mv.labels.items()]
                label_str = "{" + ",".join(label_pairs) + "}"
            
            line = f"{self.name}{label_str} {mv.value}"
            if mv.timestamp:
                line += f" {int(mv.timestamp * 1000)}"
            lines.append(line)
        
        return "\n".join(lines)


class Counter:
    """Prometheus Counter metric."""
    
    def __init__(self, name: str, help: str, labels: Optional[List[str]] = None):
        self.name = name
        self.help = help
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()
    
    def _key(self, labels: Dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))
    
    def inc(self, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment the counter."""
        labels = labels or {}
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        labels = labels or {}
        key = self._key(labels)
        return self._values.get(key, 0)
    
    def to_metric(self) -> Metric:
        """Convert to Metric object."""
        values = []
        for key, value in self._values.items():
            labels = dict(key)
            values.append(MetricValue(value=value, labels=labels))
        return Metric(name=self.name, type=MetricType.COUNTER, help=self.help, values=values)


class Gauge:
    """Prometheus Gauge metric."""
    
    def __init__(self, name: str, help: str, labels: Optional[List[str]] = None):
        self.name = name
        self.help = help
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()
    
    def _key(self, labels: Dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set the gauge value."""
        labels = labels or {}
        key = self._key(labels)
        with self._lock:
            self._values[key] = value
    
    def inc(self, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment the gauge."""
        labels = labels or {}
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value
    
    def dec(self, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Decrement the gauge."""
        labels = labels or {}
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) - value
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        labels = labels or {}
        key = self._key(labels)
        return self._values.get(key, 0)
    
    def to_metric(self) -> Metric:
        """Convert to Metric object."""
        values = []
        for key, value in self._values.items():
            labels = dict(key)
            values.append(MetricValue(value=value, labels=labels))
        return Metric(name=self.name, type=MetricType.GAUGE, help=self.help, values=values)


class Histogram:
    """Prometheus Histogram metric."""
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
    
    def __init__(self, name: str, help: str, labels: Optional[List[str]] = None,
                 buckets: Optional[tuple] = None):
        self.name = name
        self.help = help
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: Dict[tuple, Dict[float, int]] = {}
        self._sums: Dict[tuple, float] = {}
        self._totals: Dict[tuple, int] = {}
        self._lock = threading.Lock()
    
    def _key(self, labels: Dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))
    
    def _init_buckets(self, key: tuple):
        """Initialize bucket counts for a label set."""
        if key not in self._counts:
            self._counts[key] = {b: 0 for b in self.buckets}
            self._sums[key] = 0
            self._totals[key] = 0
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        labels = labels or {}
        key = self._key(labels)
        with self._lock:
            self._init_buckets(key)
            self._sums[key] += value
            self._totals[key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1
    
    def to_metric(self) -> List[Metric]:
        """Convert to list of Metric objects (bucket, sum, count)."""
        metrics = []
        
        # Bucket metric
        bucket_values = []
        for key, counts in self._counts.items():
            labels = dict(key)
            for bucket, count in counts.items():
                bucket_labels = {**labels, "le": str(bucket) if bucket != float('inf') else "+Inf"}
                bucket_values.append(MetricValue(value=count, labels=bucket_labels))
        metrics.append(Metric(
            name=f"{self.name}_bucket",
            type=MetricType.HISTOGRAM,
            help=self.help,
            values=bucket_values
        ))
        
        # Sum metric
        sum_values = []
        for key, total in self._sums.items():
            labels = dict(key)
            sum_values.append(MetricValue(value=total, labels=labels))
        metrics.append(Metric(
            name=f"{self.name}_sum",
            type=MetricType.GAUGE,
            help=f"Sum of {self.help}",
            values=sum_values
        ))
        
        # Count metric
        count_values = []
        for key, total in self._totals.items():
            labels = dict(key)
            count_values.append(MetricValue(value=total, labels=labels))
        metrics.append(Metric(
            name=f"{self.name}_count",
            type=MetricType.GAUGE,
            help=f"Count of {self.help}",
            values=count_values
        ))
        
        return metrics


class MetricsRegistry:
    """Registry for all Prometheus metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._collectors: List[Callable[[], List[Metric]]] = []
        self._lock = threading.Lock()
    
    def counter(self, name: str, help: str, labels: Optional[List[str]] = None) -> Counter:
        """Create and register a Counter metric."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            counter = Counter(name, help, labels)
            self._metrics[name] = counter
            return counter
    
    def gauge(self, name: str, help: str, labels: Optional[List[str]] = None) -> Gauge:
        """Create and register a Gauge metric."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            gauge = Gauge(name, help, labels)
            self._metrics[name] = gauge
            return gauge
    
    def histogram(self, name: str, help: str, labels: Optional[List[str]] = None,
                  buckets: Optional[tuple] = None) -> Histogram:
        """Create and register a Histogram metric."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            histogram = Histogram(name, help, labels, buckets)
            self._metrics[name] = histogram
            return histogram
    
    def register_collector(self, collector: Callable[[], List[Metric]]):
        """Register a custom metric collector."""
        self._collectors.append(collector)
    
    def collect(self) -> List[Metric]:
        """Collect all metrics."""
        metrics = []
        
        # Collect registered metrics
        for metric in self._metrics.values():
            result = metric.to_metric()
            if isinstance(result, list):
                metrics.extend(result)
            else:
                metrics.append(result)
        
        # Collect from custom collectors
        for collector in self._collectors:
            try:
                metrics.extend(collector())
            except Exception:
                pass
        
        return metrics
    
    def exposition(self) -> str:
        """Generate Prometheus exposition format output."""
        metrics = self.collect()
        lines = []
        for metric in metrics:
            lines.append(metric.to_prometheus())
        return "\n\n".join(lines) + "\n"


class MetaWatchdogMetrics:
    """Pre-defined metrics for Meta-Watchdog."""
    
    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()
        
        # Prediction metrics
        self.predictions_total = self.registry.counter(
            "meta_watchdog_predictions_total",
            "Total number of predictions made",
            labels=["model_type"]
        )
        
        self.prediction_errors_total = self.registry.counter(
            "meta_watchdog_prediction_errors_total",
            "Total number of prediction errors",
            labels=["error_type"]
        )
        
        # Reliability metrics
        self.reliability_score = self.registry.gauge(
            "meta_watchdog_reliability_score",
            "Current reliability score",
            labels=["component"]
        )
        
        self.confidence_score = self.registry.gauge(
            "meta_watchdog_confidence_score",
            "Current confidence score"
        )
        
        # Drift metrics
        self.drift_detected = self.registry.gauge(
            "meta_watchdog_drift_detected",
            "Whether drift is currently detected (1=yes, 0=no)"
        )
        
        self.drift_score = self.registry.gauge(
            "meta_watchdog_drift_score",
            "Current drift score"
        )
        
        # Performance metrics
        self.prediction_latency = self.registry.histogram(
            "meta_watchdog_prediction_latency_seconds",
            "Prediction latency in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        )
        
        self.batch_size = self.registry.histogram(
            "meta_watchdog_batch_size",
            "Batch size distribution",
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)
        )
        
        # Alert metrics
        self.alerts_total = self.registry.counter(
            "meta_watchdog_alerts_total",
            "Total number of alerts fired",
            labels=["severity", "alert_type"]
        )
        
        # Self-healing metrics
        self.healing_actions_total = self.registry.counter(
            "meta_watchdog_healing_actions_total",
            "Total number of self-healing actions taken",
            labels=["action_type"]
        )
        
        self.healing_success_total = self.registry.counter(
            "meta_watchdog_healing_success_total",
            "Total number of successful healing actions",
            labels=["action_type"]
        )
        
        # System metrics
        self.uptime_seconds = self.registry.gauge(
            "meta_watchdog_uptime_seconds",
            "System uptime in seconds"
        )
        
        self._start_time = time.time()
    
    def update_uptime(self):
        """Update the uptime metric."""
        self.uptime_seconds.set(time.time() - self._start_time)
    
    def record_prediction(self, model_type: str = "default", latency: float = 0.0):
        """Record a prediction."""
        self.predictions_total.inc(labels={"model_type": model_type})
        if latency > 0:
            self.prediction_latency.observe(latency)
    
    def record_error(self, error_type: str = "unknown"):
        """Record a prediction error."""
        self.prediction_errors_total.inc(labels={"error_type": error_type})
    
    def update_reliability(self, score: float, component: str = "overall"):
        """Update reliability score."""
        self.reliability_score.set(score, labels={"component": component})
    
    def update_confidence(self, score: float):
        """Update confidence score."""
        self.confidence_score.set(score)
    
    def update_drift(self, detected: bool, score: float = 0.0):
        """Update drift detection status."""
        self.drift_detected.set(1 if detected else 0)
        self.drift_score.set(score)
    
    def record_alert(self, severity: str, alert_type: str):
        """Record an alert."""
        self.alerts_total.inc(labels={"severity": severity, "alert_type": alert_type})
    
    def record_healing(self, action_type: str, success: bool):
        """Record a healing action."""
        self.healing_actions_total.inc(labels={"action_type": action_type})
        if success:
            self.healing_success_total.inc(labels={"action_type": action_type})


class PrometheusExporter:
    """HTTP server that exports metrics in Prometheus format."""
    
    def __init__(self, registry: MetricsRegistry, port: int = 9090, host: str = "0.0.0.0"):
        self.registry = registry
        self.port = port
        self.host = host
        self._server: Optional[socketserver.TCPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def _create_handler(self):
        """Create HTTP request handler."""
        registry = self.registry
        
        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    content = registry.exposition()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content.encode())
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        return MetricsHandler
    
    def start(self, background: bool = True):
        """Start the exporter server."""
        handler = self._create_handler()
        self._server = socketserver.TCPServer((self.host, self.port), handler)
        
        if background:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"Prometheus exporter started on http://{self.host}:{self.port}/metrics")
        else:
            print(f"Prometheus exporter serving on http://{self.host}:{self.port}/metrics")
            self._server.serve_forever()
    
    def stop(self):
        """Stop the exporter server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None


# Global registry instance
default_registry = MetricsRegistry()


def get_default_registry() -> MetricsRegistry:
    """Get the default metrics registry."""
    return default_registry


__all__ = [
    "MetricType",
    "MetricValue",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "MetaWatchdogMetrics",
    "PrometheusExporter",
    "default_registry",
    "get_default_registry",
]
