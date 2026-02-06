"""
Tests for the benchmarking module.
"""

import pytest
import time

from meta_watchdog.benchmarks import (
    Benchmark,
    BenchmarkResult,
    Profiler,
    Timer,
    TimingResult,
    get_profiler,
    get_timer,
    timed,
)


class TestTimer:
    """Tests for Timer class."""
    
    def test_basic_timing(self):
        """Test basic start/stop timing."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)  # 10ms
        elapsed = timer.stop()
        
        assert elapsed >= 10  # At least 10ms
        assert elapsed < 100  # Less than 100ms (reasonable margin)
    
    def test_elapsed_property(self):
        """Test elapsed time property."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        
        # Should work without stopping
        elapsed = timer.elapsed_ms
        assert elapsed >= 10
    
    def test_context_manager(self):
        """Test timer as context manager."""
        timer = Timer()
        
        with timer.measure("test_operation"):
            time.sleep(0.01)
        
        timings = timer.get_timings()
        assert len(timings) == 1
        assert timings[0].name == "test_operation"
        assert timings[0].duration_ms >= 10
    
    def test_multiple_measurements(self):
        """Test multiple timing measurements."""
        timer = Timer()
        
        for i in range(3):
            with timer.measure(f"op_{i}"):
                time.sleep(0.005)
        
        timings = timer.get_timings()
        assert len(timings) == 3
    
    def test_clear(self):
        """Test clearing timer data."""
        timer = Timer()
        
        with timer.measure("test"):
            pass
        
        timer.clear()
        
        assert len(timer.get_timings()) == 0


class TestTimedDecorator:
    """Tests for timed decorator."""
    
    def test_timed_decorator(self, caplog):
        """Test function timing with decorator."""
        @timed("custom_name")
        def slow_function():
            time.sleep(0.01)
            return "result"
        
        result = slow_function()
        
        assert result == "result"
    
    def test_timed_decorator_default_name(self):
        """Test decorator with default function name."""
        @timed()
        def my_function():
            return 42
        
        result = my_function()
        assert result == 42


class TestBenchmark:
    """Tests for Benchmark class."""
    
    def test_basic_benchmark(self):
        """Test basic benchmark execution."""
        def simple_function():
            return sum(range(100))
        
        benchmark = Benchmark(warmup_iterations=2, min_iterations=5)
        result = benchmark.run(simple_function, name="sum_test", iterations=10)
        
        assert isinstance(result, BenchmarkResult)
        assert result.name == "sum_test"
        assert result.iterations == 10
        assert result.mean_time_ms > 0
        assert result.throughput > 0
    
    def test_benchmark_statistics(self):
        """Test benchmark statistics calculation."""
        def variable_function():
            time.sleep(0.001)
        
        benchmark = Benchmark(warmup_iterations=1)
        result = benchmark.run(variable_function, iterations=10)
        
        assert result.min_time_ms <= result.mean_time_ms
        assert result.mean_time_ms <= result.max_time_ms
        assert result.p95_time_ms <= result.p99_time_ms
    
    def test_benchmark_with_args(self):
        """Test benchmark with function arguments."""
        def add_numbers(a, b):
            return a + b
        
        benchmark = Benchmark(warmup_iterations=1)
        result = benchmark.run(add_numbers, args=(10, 20), iterations=10)
        
        assert result.iterations == 10
    
    def test_benchmark_with_kwargs(self):
        """Test benchmark with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        benchmark = Benchmark(warmup_iterations=1)
        result = benchmark.run(
            greet,
            kwargs={"name": "World", "greeting": "Hi"},
            iterations=10
        )
        
        assert result.iterations == 10
    
    def test_benchmark_results_collection(self):
        """Test collecting benchmark results."""
        benchmark = Benchmark(warmup_iterations=1)
        
        benchmark.run(lambda: None, name="test1", iterations=5)
        benchmark.run(lambda: None, name="test2", iterations=5)
        
        results = benchmark.get_results()
        assert len(results) == 2
    
    def test_benchmark_comparison(self):
        """Test comparing benchmark results."""
        benchmark = Benchmark(warmup_iterations=1)
        
        baseline = benchmark.run(lambda: time.sleep(0.001), name="baseline", iterations=5)
        faster = benchmark.run(lambda: None, name="faster", iterations=5)
        
        comparison = benchmark.compare(baseline, faster)
        
        assert "mean_diff_ms" in comparison
        assert "mean_diff_pct" in comparison
        assert "improvement" in comparison
    
    def test_benchmark_to_dict(self):
        """Test benchmark result serialization."""
        benchmark = Benchmark(warmup_iterations=1)
        result = benchmark.run(lambda: None, name="serialize_test", iterations=5)
        
        data = result.to_dict()
        
        assert data["name"] == "serialize_test"
        assert data["iterations"] == 5
        assert "mean_time_ms" in data
        assert "timestamp" in data
    
    def test_benchmark_str(self):
        """Test benchmark result string representation."""
        benchmark = Benchmark(warmup_iterations=1)
        result = benchmark.run(lambda: None, name="str_test", iterations=5)
        
        string = str(result)
        
        assert "str_test" in string
        assert "Iterations" in string
        assert "Mean" in string


class TestProfiler:
    """Tests for Profiler class."""
    
    def test_basic_profiling(self):
        """Test basic profiling."""
        profiler = Profiler()
        
        with profiler.profile("test_section"):
            time.sleep(0.01)
        
        stats = profiler.get_stats()
        
        assert "test_section" in stats
        assert stats["test_section"]["call_count"] == 1
        assert stats["test_section"]["total_time_ms"] >= 10
    
    def test_multiple_calls(self):
        """Test profiling multiple calls."""
        profiler = Profiler()
        
        for _ in range(5):
            with profiler.profile("repeated"):
                pass
        
        stats = profiler.get_stats()
        
        assert stats["repeated"]["call_count"] == 5
    
    def test_enable_disable(self):
        """Test enabling/disabling profiler."""
        profiler = Profiler()
        profiler.disable()
        
        with profiler.profile("disabled_section"):
            pass
        
        stats = profiler.get_stats()
        assert "disabled_section" not in stats
        
        profiler.enable()
        
        with profiler.profile("enabled_section"):
            pass
        
        stats = profiler.get_stats()
        assert "enabled_section" in stats
    
    def test_reset(self):
        """Test resetting profiler data."""
        profiler = Profiler()
        
        with profiler.profile("test"):
            pass
        
        assert len(profiler.get_stats()) > 0
        
        profiler.reset()
        
        assert len(profiler.get_stats()) == 0
    
    def test_report(self):
        """Test profiling report generation."""
        profiler = Profiler()
        
        with profiler.profile("section_a"):
            time.sleep(0.005)
        
        with profiler.profile("section_b"):
            time.sleep(0.001)
        
        report = profiler.report()
        
        assert "section_a" in report
        assert "section_b" in report
        assert "Profiling Report" in report
    
    def test_empty_report(self):
        """Test report with no data."""
        profiler = Profiler()
        report = profiler.report()
        
        assert "No profiling data" in report


class TestGlobalInstances:
    """Tests for global timer and profiler instances."""
    
    def test_get_timer(self):
        """Test getting global timer."""
        timer = get_timer()
        assert isinstance(timer, Timer)
    
    def test_get_profiler(self):
        """Test getting global profiler."""
        profiler = get_profiler()
        assert isinstance(profiler, Profiler)
    
    def test_global_instances_persistence(self):
        """Test that global instances persist."""
        timer1 = get_timer()
        timer2 = get_timer()
        assert timer1 is timer2
        
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        assert profiler1 is profiler2


class TestTimingResult:
    """Tests for TimingResult dataclass."""
    
    def test_timing_result_creation(self):
        """Test creating a timing result."""
        result = TimingResult(
            name="test",
            duration_ms=42.5,
            metadata={"key": "value"}
        )
        
        assert result.name == "test"
        assert result.duration_ms == 42.5
        assert result.metadata["key"] == "value"
        assert result.timestamp is not None
