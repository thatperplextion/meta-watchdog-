"""
REST API Server for Meta-Watchdog.

This module provides a simple HTTP API for remote monitoring
and integration with external systems.
"""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standard API response."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class MetaWatchdogAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Meta-Watchdog API."""
    
    # Class-level reference to the orchestrator
    orchestrator = None
    api_key: Optional[str] = None
    
    def log_message(self, format: str, *args) -> None:
        """Override to use logging module."""
        logger.info(f"API: {format % args}")
    
    def _send_json_response(
        self,
        response: APIResponse,
        status_code: int = 200
    ) -> None:
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()
        self.wfile.write(response.to_json().encode())
    
    def _check_auth(self) -> bool:
        """Check API key authentication."""
        if not self.api_key:
            return True
        
        provided_key = self.headers.get("X-API-Key")
        return provided_key == self.api_key
    
    def _parse_request_body(self) -> Optional[Dict[str, Any]]:
        """Parse JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                return json.loads(body.decode())
            return {}
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            return None
    
    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        if not self._check_auth():
            self._send_json_response(
                APIResponse(success=False, error="Unauthorized"),
                status_code=401
            )
            return
        
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # Route handling
        routes = {
            "/": self._handle_root,
            "/health": self._handle_health,
            "/status": self._handle_status,
            "/metrics": self._handle_metrics,
            "/reliability": self._handle_reliability,
            "/alerts": self._handle_alerts,
            "/predictions": self._handle_predictions,
            "/analysis": self._handle_analysis,
        }
        
        handler = routes.get(path)
        if handler:
            try:
                handler(query)
            except Exception as e:
                logger.error(f"API error: {e}")
                self._send_json_response(
                    APIResponse(success=False, error=str(e)),
                    status_code=500
                )
        else:
            self._send_json_response(
                APIResponse(success=False, error="Not found"),
                status_code=404
            )
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        if not self._check_auth():
            self._send_json_response(
                APIResponse(success=False, error="Unauthorized"),
                status_code=401
            )
            return
        
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._parse_request_body()
        
        if body is None:
            self._send_json_response(
                APIResponse(success=False, error="Invalid JSON body"),
                status_code=400
            )
            return
        
        routes = {
            "/predict": self._handle_predict,
            "/alerts/resolve": self._handle_resolve_alert,
            "/config": self._handle_update_config,
        }
        
        handler = routes.get(path)
        if handler:
            try:
                handler(body)
            except Exception as e:
                logger.error(f"API error: {e}")
                self._send_json_response(
                    APIResponse(success=False, error=str(e)),
                    status_code=500
                )
        else:
            self._send_json_response(
                APIResponse(success=False, error="Not found"),
                status_code=404
            )
    
    # GET handlers
    def _handle_root(self, query: Dict) -> None:
        """Handle root endpoint."""
        self._send_json_response(APIResponse(
            success=True,
            data={
                "service": "Meta-Watchdog API",
                "version": "1.0.0",
                "endpoints": [
                    "/health",
                    "/status",
                    "/metrics",
                    "/reliability",
                    "/alerts",
                    "/predictions",
                    "/analysis",
                ]
            }
        ))
    
    def _handle_health(self, query: Dict) -> None:
        """Handle health check."""
        self._send_json_response(APIResponse(
            success=True,
            data={"status": "healthy", "service": "meta-watchdog"}
        ))
    
    def _handle_status(self, query: Dict) -> None:
        """Handle status endpoint."""
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        snapshot = self.orchestrator.get_health_snapshot()
        self._send_json_response(APIResponse(
            success=True,
            data={
                "status": snapshot.status.value if hasattr(snapshot, 'status') else "unknown",
                "reliability_score": getattr(snapshot, 'reliability_score', 0),
                "failure_probability": getattr(snapshot, 'failure_probability', 0),
                "predictions_count": getattr(snapshot, 'total_predictions', 0),
            }
        ))
    
    def _handle_metrics(self, query: Dict) -> None:
        """Handle metrics endpoint."""
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        metrics = self.orchestrator.get_current_metrics()
        self._send_json_response(APIResponse(
            success=True,
            data=metrics if isinstance(metrics, dict) else {"raw": str(metrics)}
        ))
    
    def _handle_reliability(self, query: Dict) -> None:
        """Handle reliability score endpoint."""
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        score = self.orchestrator.get_reliability_score()
        self._send_json_response(APIResponse(
            success=True,
            data={
                "score": score.overall if hasattr(score, 'overall') else float(score),
                "components": score.components if hasattr(score, 'components') else {},
            }
        ))
    
    def _handle_alerts(self, query: Dict) -> None:
        """Handle alerts endpoint."""
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        alerts = self.orchestrator.get_active_alerts()
        self._send_json_response(APIResponse(
            success=True,
            data={
                "count": len(alerts),
                "alerts": [a.to_dict() if hasattr(a, 'to_dict') else str(a) for a in alerts]
            }
        ))
    
    def _handle_predictions(self, query: Dict) -> None:
        """Handle predictions history endpoint."""
        limit = int(query.get("limit", [100])[0])
        
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        predictions = self.orchestrator.get_prediction_history(limit)
        self._send_json_response(APIResponse(
            success=True,
            data={
                "count": len(predictions),
                "predictions": predictions
            }
        ))
    
    def _handle_analysis(self, query: Dict) -> None:
        """Handle full analysis endpoint."""
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        analysis = self.orchestrator.get_full_analysis()
        self._send_json_response(APIResponse(
            success=True,
            data=analysis.to_dict() if hasattr(analysis, 'to_dict') else str(analysis)
        ))
    
    # POST handlers
    def _handle_predict(self, body: Dict) -> None:
        """Handle prediction request."""
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        features = body.get("features")
        if not features:
            self._send_json_response(APIResponse(
                success=False, error="Missing 'features' in request body"
            ), status_code=400)
            return
        
        import numpy as np
        X = np.array(features)
        
        result = self.orchestrator.predict(X)
        self._send_json_response(APIResponse(
            success=True,
            data={
                "predictions": result.predictions.tolist() if hasattr(result.predictions, 'tolist') else result.predictions,
                "confidence": result.confidence.tolist() if hasattr(result.confidence, 'tolist') else result.confidence,
            }
        ))
    
    def _handle_resolve_alert(self, body: Dict) -> None:
        """Handle alert resolution."""
        alert_id = body.get("alert_id")
        if not alert_id:
            self._send_json_response(APIResponse(
                success=False, error="Missing 'alert_id' in request body"
            ), status_code=400)
            return
        
        if self.orchestrator and hasattr(self.orchestrator, 'resolve_alert'):
            success = self.orchestrator.resolve_alert(alert_id)
            self._send_json_response(APIResponse(
                success=success,
                data={"alert_id": alert_id, "resolved": success}
            ))
        else:
            self._send_json_response(APIResponse(
                success=False, error="Alert resolution not available"
            ), status_code=503)
    
    def _handle_update_config(self, body: Dict) -> None:
        """Handle configuration update."""
        if not self.orchestrator:
            self._send_json_response(APIResponse(
                success=False, error="Orchestrator not initialized"
            ), status_code=503)
            return
        
        # Apply configuration updates
        if hasattr(self.orchestrator, 'update_config'):
            self.orchestrator.update_config(body)
            self._send_json_response(APIResponse(
                success=True,
                data={"message": "Configuration updated"}
            ))
        else:
            self._send_json_response(APIResponse(
                success=False, error="Configuration update not supported"
            ), status_code=501)


class MetaWatchdogAPIServer:
    """API server for Meta-Watchdog."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        orchestrator: Any = None,
        api_key: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.orchestrator = orchestrator
        self.api_key = api_key
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self, background: bool = True) -> None:
        """Start the API server."""
        # Configure handler
        MetaWatchdogAPIHandler.orchestrator = self.orchestrator
        MetaWatchdogAPIHandler.api_key = self.api_key
        
        self._server = HTTPServer((self.host, self.port), MetaWatchdogAPIHandler)
        
        logger.info(f"Starting API server on {self.host}:{self.port}")
        
        if background:
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
        else:
            self._server.serve_forever()
    
    def stop(self) -> None:
        """Stop the API server."""
        if self._server:
            self._server.shutdown()
            logger.info("API server stopped")
    
    def set_orchestrator(self, orchestrator: Any) -> None:
        """Update the orchestrator reference."""
        self.orchestrator = orchestrator
        MetaWatchdogAPIHandler.orchestrator = orchestrator


def create_api_server(
    orchestrator: Any = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    api_key: Optional[str] = None,
) -> MetaWatchdogAPIServer:
    """Factory function to create API server."""
    return MetaWatchdogAPIServer(
        host=host,
        port=port,
        orchestrator=orchestrator,
        api_key=api_key,
    )
