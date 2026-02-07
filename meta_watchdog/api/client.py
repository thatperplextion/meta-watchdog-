"""
Meta-Watchdog API Client

A Python client library for interacting with the Meta-Watchdog REST API.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import time
import urllib.request
import urllib.error
import urllib.parse


class ClientError(Exception):
    """Base exception for client errors."""
    pass


class ConnectionError(ClientError):
    """Failed to connect to the server."""
    pass


class APIError(ClientError):
    """Server returned an error response."""
    def __init__(self, message: str, status_code: int, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class TimeoutError(ClientError):
    """Request timed out."""
    pass


@dataclass
class HealthStatus:
    """Health check response."""
    status: str
    timestamp: str
    uptime: float
    version: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> "HealthStatus":
        return cls(
            status=data.get("status", "unknown"),
            timestamp=data.get("timestamp", ""),
            uptime=data.get("uptime", 0.0),
            version=data.get("version", "unknown")
        )


@dataclass
class SystemStatus:
    """System status response."""
    total_predictions: int
    total_batches: int
    reliability_score: float
    confidence: float
    drift_detected: bool
    model_health: str
    last_update: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SystemStatus":
        return cls(
            total_predictions=data.get("total_predictions", 0),
            total_batches=data.get("total_batches", 0),
            reliability_score=data.get("reliability_score", 0.0),
            confidence=data.get("confidence", 0.0),
            drift_detected=data.get("drift_detected", False),
            model_health=data.get("model_health", "unknown"),
            last_update=data.get("last_update", "")
        )


@dataclass
class MetricsSummary:
    """Metrics summary response."""
    timestamp: str
    reliability: Dict[str, float]
    drift: Dict[str, Any]
    performance: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MetricsSummary":
        return cls(
            timestamp=data.get("timestamp", ""),
            reliability=data.get("reliability", {}),
            drift=data.get("drift", {}),
            performance=data.get("performance", {})
        )


@dataclass
class ReliabilityReport:
    """Reliability report response."""
    overall_score: float
    components: Dict[str, float]
    history: List[Dict[str, Any]]
    recommendations: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ReliabilityReport":
        return cls(
            overall_score=data.get("overall_score", 0.0),
            components=data.get("components", {}),
            history=data.get("history", []),
            recommendations=data.get("recommendations", [])
        )


@dataclass
class Alert:
    """Alert data."""
    id: str
    severity: str
    message: str
    timestamp: str
    acknowledged: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Alert":
        return cls(
            id=data.get("id", ""),
            severity=data.get("severity", "info"),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", ""),
            acknowledged=data.get("acknowledged", False),
            context=data.get("context", {})
        )


class MetaWatchdogClient:
    """
    Client for the Meta-Watchdog REST API.
    
    Example:
        client = MetaWatchdogClient("http://localhost:8080")
        health = client.health()
        print(f"Status: {health.status}")
        
        status = client.status()
        print(f"Reliability: {status.reliability_score}")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the Meta-Watchdog API
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._session_headers: Dict[str, str] = {}
    
    def _build_url(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build full URL with optional query parameters."""
        url = f"{self.base_url}{path}"
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        return url
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MetaWatchdogClient/1.0"
        }
        headers.update(self._session_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request with retries."""
        url = self._build_url(path, params)
        headers = self._build_headers()
        
        body = None
        if data:
            body = json.dumps(data).encode("utf-8")
        
        last_error = None
        for attempt in range(self.retry_count):
            try:
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers=headers,
                    method=method
                )
                
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    content = response.read().decode("utf-8")
                    if content:
                        return json.loads(content)
                    return {}
            
            except urllib.error.HTTPError as e:
                content = e.read().decode("utf-8")
                try:
                    error_data = json.loads(content)
                except json.JSONDecodeError:
                    error_data = {"message": content}
                raise APIError(
                    f"HTTP {e.code}: {e.reason}",
                    status_code=e.code,
                    response=error_data
                )
            
            except urllib.error.URLError as e:
                last_error = ConnectionError(f"Failed to connect: {e.reason}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise last_error
            
            except TimeoutError:
                last_error = TimeoutError(f"Request timed out after {self.timeout}s")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise last_error
        
        if last_error:
            raise last_error
        raise ClientError("Unknown error")
    
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        return self._request("GET", path, params=params)
    
    def _post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request."""
        return self._request("POST", path, data=data)
    
    def _put(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a PUT request."""
        return self._request("PUT", path, data=data)
    
    def _delete(self, path: str) -> Dict[str, Any]:
        """Make a DELETE request."""
        return self._request("DELETE", path)
    
    # Health and status endpoints
    
    def health(self) -> HealthStatus:
        """Check API health status."""
        data = self._get("/health")
        return HealthStatus.from_dict(data)
    
    def status(self) -> SystemStatus:
        """Get system status."""
        data = self._get("/status")
        return SystemStatus.from_dict(data)
    
    def is_healthy(self) -> bool:
        """Quick health check."""
        try:
            health = self.health()
            return health.status == "healthy"
        except ClientError:
            return False
    
    # Metrics endpoints
    
    def metrics(self) -> MetricsSummary:
        """Get current metrics summary."""
        data = self._get("/metrics")
        return MetricsSummary.from_dict(data)
    
    def metrics_history(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get metrics history."""
        params = {"limit": limit}
        if metric_name:
            params["metric"] = metric_name
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time
        
        data = self._get("/metrics/history", params=params)
        return data.get("history", [])
    
    # Reliability endpoints
    
    def reliability(self) -> ReliabilityReport:
        """Get reliability report."""
        data = self._get("/reliability")
        return ReliabilityReport.from_dict(data)
    
    def reliability_score(self) -> float:
        """Get current reliability score."""
        report = self.reliability()
        return report.overall_score
    
    # Alert endpoints
    
    def alerts(
        self,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 50
    ) -> List[Alert]:
        """Get alerts."""
        params: Dict[str, Any] = {"limit": limit}
        if severity:
            params["severity"] = severity
        if acknowledged is not None:
            params["acknowledged"] = str(acknowledged).lower()
        
        data = self._get("/alerts", params=params)
        return [Alert.from_dict(a) for a in data.get("alerts", [])]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            self._post(f"/alerts/{alert_id}/acknowledge")
            return True
        except APIError:
            return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert."""
        try:
            self._delete(f"/alerts/{alert_id}")
            return True
        except APIError:
            return False
    
    # Drift detection endpoints
    
    def drift_status(self) -> Dict[str, Any]:
        """Get drift detection status."""
        return self._get("/drift")
    
    def is_drifting(self) -> bool:
        """Check if drift is detected."""
        status = self.drift_status()
        return status.get("detected", False)
    
    # Model management endpoints
    
    def models(self) -> List[Dict[str, Any]]:
        """Get list of monitored models."""
        data = self._get("/models")
        return data.get("models", [])
    
    def model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        return self._get(f"/models/{model_id}")
    
    # Configuration endpoints
    
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._get("/config")
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration."""
        return self._put("/config", data=config)
    
    # Action endpoints
    
    def trigger_healing(self, action: str = "auto") -> Dict[str, Any]:
        """Trigger a healing action."""
        return self._post("/actions/heal", data={"action": action})
    
    def trigger_checkpoint(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a checkpoint."""
        data = {}
        if name:
            data["name"] = name
        return self._post("/actions/checkpoint", data=data)
    
    # Convenience methods
    
    def wait_for_healthy(self, timeout: float = 60.0, interval: float = 1.0) -> bool:
        """Wait for the system to become healthy."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_healthy():
                return True
            time.sleep(interval)
        return False
    
    def summary(self) -> Dict[str, Any]:
        """Get a complete system summary."""
        return {
            "health": self.health().__dict__,
            "status": self.status().__dict__,
            "reliability": self.reliability().__dict__,
            "drift_detected": self.is_drifting()
        }


class AsyncMetaWatchdogClient:
    """
    Async client placeholder for future async support.
    
    This is a placeholder for async/await support using aiohttp.
    Currently returns synchronous client operations.
    """
    
    def __init__(self, *args, **kwargs):
        self._sync_client = MetaWatchdogClient(*args, **kwargs)
        print("Warning: AsyncMetaWatchdogClient is using synchronous operations. "
              "Install aiohttp for true async support.")
    
    async def health(self) -> HealthStatus:
        return self._sync_client.health()
    
    async def status(self) -> SystemStatus:
        return self._sync_client.status()
    
    async def metrics(self) -> MetricsSummary:
        return self._sync_client.metrics()
    
    async def reliability(self) -> ReliabilityReport:
        return self._sync_client.reliability()
    
    async def alerts(self, **kwargs) -> List[Alert]:
        return self._sync_client.alerts(**kwargs)


def create_client(
    base_url: str = "http://localhost:8080",
    **kwargs
) -> MetaWatchdogClient:
    """
    Factory function to create a client.
    
    Args:
        base_url: Base URL of the Meta-Watchdog API
        **kwargs: Additional arguments passed to MetaWatchdogClient
    
    Returns:
        Configured MetaWatchdogClient instance
    """
    return MetaWatchdogClient(base_url, **kwargs)


__all__ = [
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
