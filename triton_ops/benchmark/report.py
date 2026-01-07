"""Performance report generation for benchmarks."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from triton_ops.models import KernelMetrics


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        kernel_name: Name of the kernel
        problem_size: Problem dimensions
        config: Kernel configuration used
        metrics: Performance metrics
        correctness: Whether output was correct
        timestamp: When benchmark was run
    """

    kernel_name: str
    problem_size: tuple
    config: Dict[str, Any]
    metrics: KernelMetrics
    correctness: bool
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ComparisonResult:
    """Result comparing Triton kernel to baseline.

    Attributes:
        kernel_name: Name of the kernel
        problem_size: Problem dimensions
        triton_metrics: Metrics for Triton implementation
        baseline_metrics: Metrics for baseline (PyTorch/cuBLAS)
        speedup: Triton speedup over baseline
        correctness: Whether outputs match
    """

    kernel_name: str
    problem_size: tuple
    triton_metrics: KernelMetrics
    baseline_metrics: KernelMetrics
    speedup: float
    correctness: bool


class PerformanceReport:
    """Generator for human-readable performance reports.

    Collects benchmark results and generates formatted reports.
    """

    def __init__(self, title: str = "Triton Operators Benchmark Report"):
        self.title = title
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []
        self.metadata: Dict[str, Any] = {}

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def add_comparison(self, comparison: ComparisonResult) -> None:
        """Add a comparison result."""
        self.comparisons.append(comparison)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set report metadata."""
        self.metadata[key] = value

    def generate_text_report(self) -> str:
        """Generate human-readable text report.

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f" {self.title}")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Metadata
        if self.metadata:
            lines.append("System Information:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Individual results
        if self.results:
            lines.append("-" * 80)
            lines.append(" Benchmark Results")
            lines.append("-" * 80)

            for result in self.results:
                lines.append(f"\n{result.kernel_name}")
                lines.append(f"  Problem size: {result.problem_size}")
                lines.append(f"  Latency: {result.metrics.latency_ms:.3f} ms")
                lines.append(f"  Throughput: {result.metrics.throughput_tflops:.2f} TFLOPS")
                lines.append(
                    f"  Bandwidth: {result.metrics.bandwidth_gbps:.1f} GB/s "
                    f"({result.metrics.bandwidth_utilization:.1f}%)"
                )
                lines.append(f"  Correct: {'✓' if result.correctness else '✗'}")
            lines.append("")

        # Comparisons
        if self.comparisons:
            lines.append("-" * 80)
            lines.append(" Comparison vs Baseline")
            lines.append("-" * 80)

            for comp in self.comparisons:
                lines.append(f"\n{comp.kernel_name}")
                lines.append(f"  Problem size: {comp.problem_size}")
                lines.append(f"  Triton:   {comp.triton_metrics.latency_ms:.3f} ms")
                lines.append(f"  Baseline: {comp.baseline_metrics.latency_ms:.3f} ms")
                lines.append(f"  Speedup:  {comp.speedup:.2f}x")
                lines.append(f"  Correct:  {'✓' if comp.correctness else '✗'}")
            lines.append("")

        # Summary
        lines.append("-" * 80)
        lines.append(" Summary")
        lines.append("-" * 80)

        total_results = len(self.results)
        correct_results = sum(1 for r in self.results if r.correctness)
        lines.append(f"Total benchmarks: {total_results}")
        lines.append(f"Correct: {correct_results}/{total_results}")

        if self.comparisons:
            avg_speedup = sum(c.speedup for c in self.comparisons) / len(self.comparisons)
            lines.append(f"Average speedup: {avg_speedup:.2f}x")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_json_report(self) -> str:
        """Generate JSON report.

        Returns:
            JSON formatted report
        """
        report: Dict[str, Any] = {
            "title": self.title,
            "generated": datetime.now().isoformat(),
            "metadata": self.metadata,
            "results": [],
            "comparisons": [],
        }

        for result in self.results:
            report["results"].append(
                {
                    "kernel_name": result.kernel_name,
                    "problem_size": result.problem_size,
                    "config": result.config,
                    "metrics": asdict(result.metrics),
                    "correctness": result.correctness,
                    "timestamp": result.timestamp,
                }
            )

        for comp in self.comparisons:
            report["comparisons"].append(
                {
                    "kernel_name": comp.kernel_name,
                    "problem_size": comp.problem_size,
                    "triton_metrics": asdict(comp.triton_metrics),
                    "baseline_metrics": asdict(comp.baseline_metrics),
                    "speedup": comp.speedup,
                    "correctness": comp.correctness,
                }
            )

        return json.dumps(report, indent=2)

    def save(self, filepath: str, format: str = "text") -> None:
        """Save report to file.

        Args:
            filepath: Output file path
            format: "text" or "json"
        """
        if format == "json":
            content = self.generate_json_report()
        else:
            content = self.generate_text_report()

        with open(filepath, "w") as f:
            f.write(content)
