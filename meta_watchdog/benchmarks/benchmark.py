"""
Benchmarking Module for Meta-Watchdog.

This module provides tools for benchmarking and profiling
the Meta-Watchdog system performance.
"""

import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import functools

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of a timed operation."""
    
    name: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class BenchmarkResult:
    """Result of a benchmark run."""
    
    name: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput: float  # operations per second
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "mean_time_ms": self.mean_time_ms,
            "std_dev_ms": self.std_dev_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "median_time_ms": self.median_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "p99_time_ms": self.p99_time_ms,
            "throughput": self.throughput,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        return (
            f"Benchmark: {self.name}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_time_ms:.3f}ms\n"
            f"  Std Dev: {self.std_dev_ms:.3f}ms\n"
            f"  Min/Max: {self.min_time_ms:.3f}ms / {self.max_time_ms:.3f}ms\n"
            f"  P95/P99: {self.p95_time_ms:.3f}ms / {self.p99_time_ms:.3f}ms\n"
            f"  Throughput: {self.throughput:.1f} ops/sec"
        )


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._timings: List[TimingResult] = []
    
    def start(self) -> None:
        """Start the timer."""
        self._start = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in ms."""
        self._end = time.perf_counter()
        return self.elapsed_ms
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start is None:
            return 0.0
        end = self._end or time.perf_counter()
        return (end - self._start) * 1000
    
    @contextmanager
    def measure(self, name: str = "operation"):
        """Context manager for measuring execution time."""
        start = time.perf_counter()
        try:
            yield self
        finally:
            duration = (time.perf_counter() - start) * 1000
            self._timings.append(TimingResult(name=name, duration_ms=duration))
    
    def get_timings(self) -> List[TimingResult]:
        """Get all recorded timings."""
        return self._timings
    
    def clear(self) -> None:
        """Clear recorded timings."""
        self._timings = []
        self._start = None
        self._end = None


def timed(name: Optional[str] = None):
    """Decorator to measure function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = (time.perf_counter() - start) * 1000
                logger.debug(f"{func_name} executed in {duration:.3f}ms")
        return wrapper
    return decorator


class Benchmark:
    """Benchmark runner for performance testing."""
    
    def __init__(
        self,
        warmup_iterations: int = 5,
        min_iterations: int = 10,
        max_iterations: int = 1000,
        target_time_seconds: float = 1.0,
    ):
        self.warmup_iterations = warmup_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.target_time_seconds = target_time_seconds
        self._results: List[BenchmarkResult] = []
    
    def run(
        self,
        func: Callable,
        name: Optional[str] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run a benchmark on a function."""
        kwargs = kwargs or {}
        benchmark_name = name or func.__name__
        
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Determine iterations
        if iterations is None:
            # Auto-calibrate based on target time
            start = time.perf_counter()
            func(*args, **kwargs)
            single_time = time.perf_counter() - start
            
            if single_time > 0:
                iterations = int(self.target_time_seconds / single_time)
                iterations = max(self.min_iterations, min(iterations, self.max_iterations))
            else:
                iterations = self.min_iterations
        
        # Run benchmark
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            timings.append(duration)
        
        # Calculate statistics
        result = self._calculate_stats(benchmark_name, timings)
        self._results.append(result)
        
        return result
    
    def _calculate_stats(
        self,
        name: str,
        timings: List[float]
    ) -> BenchmarkResult:
        """Calculate benchmark statistics."""
        sorted_timings = sorted(timings)
        n = len(timings)
        
        total = sum(timings)
        mean = statistics.mean(timings)
        std_dev = statistics.stdev(timings) if n > 1 else 0.0
        
        # Percentiles
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        return BenchmarkResult(
            name=name,
            iterations=n,
            total_time_ms=total,
            mean_time_ms=mean,
            std_dev_ms=std_dev,
            min_time_ms=min(timings),
            max_time_ms=max(timings),
            median_time_ms=statistics.median(timings),
            p95_time_ms=sorted_timings[p95_idx] if p95_idx < n else sorted_timings[-1],
            p99_time_ms=sorted_timings[p99_idx] if p99_idx < n else sorted_timings[-1],
            throughput=1000 / mean if mean > 0 else 0,
        )
    
    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results
    
    def clear_results(self) -> None:
        """Clear benchmark results."""
        self._results = []
    
    def compare(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult
    ) -> Dict[str, Any]:
        """Compare two benchmark results."""
        mean_diff = current.mean_time_ms - baseline.mean_time_ms
        mean_pct = (mean_diff / baseline.mean_time_ms * 100) if baseline.mean_time_ms > 0 else 0
        
        throughput_diff = current.throughput - baseline.throughput
        throughput_pct = (throughput_diff / baseline.throughput * 100) if baseline.throughput > 0 else 0
        
        return {
            "baseline": baseline.name,
            "current": current.name,
            "mean_diff_ms": mean_diff,
            "mean_diff_pct": mean_pct,
            "throughput_diff": throughput_diff,
            "throughput_diff_pct": throughput_pct,
            "regression": mean_pct > 10,  # >10% slower is a regression
            "improvement": mean_pct < -10,  # >10% faster is an improvement
        }


class Profiler:
    """Simple profiler for tracking execution patterns."""
    
    def __init__(self):
        self._call_counts: Dict[str, int] = {}
        self._total_times: Dict[str, float] = {}
        self._enabled = True
    
    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable profiling."""
        self._enabled = False
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code section."""
        if not self._enabled:
            yield
            return
        
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000
            self._call_counts[name] = self._call_counts.get(name, 0) + 1
            self._total_times[name] = self._total_times.get(name, 0) + duration
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        stats = {}
        for name in self._call_counts:
            count = self._call_counts[name]
            total = self._total_times[name]
            stats[name] = {
                "call_count": count,
                "total_time_ms": total,
                "avg_time_ms": total / count if count > 0 else 0,
            }
        return stats
    
    def reset(self) -> None:
        """Reset profiling data."""
        self._call_counts = {}
        self._total_times = {}
    
    def report(self) -> str:
        """Generate a profiling report."""
        stats = self.get_stats()
        if not stats:
            return "No profiling data"
        
        lines = ["Profiling Report", "=" * 60]
        
        # Sort by total time
        sorted_stats = sorted(
            stats.items(),
            key=lambda x: x[1]["total_time_ms"],
            reverse=True
        )
        
        for name, data in sorted_stats:
            lines.append(
                f"{name:40s} "
                f"calls={data['call_count']:6d} "
                f"total={data['total_time_ms']:10.2f}ms "
                f"avg={data['avg_time_ms']:8.2f}ms"
            )
        
        return "\n".join(lines)


# Global instances for convenience
_default_timer = Timer()
_default_profiler = Profiler()


def get_timer() -> Timer:
    """Get the default timer instance."""
    return _default_timer


def get_profiler() -> Profiler:
    """Get the default profiler instance."""
    return _default_profiler
