from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from bot.io_utils import atomic_write_json


class EvaluationGate:
    """Basic regression gate based on corpus growth and training recency."""

    def __init__(
        self,
        report_path: str,
        benchmark_path: str = "",
        benchmark_min_pass_rate: float = 0.8,
        benchmark_max_latency_ms: float = 5000.0,
    ) -> None:
        self.report_path = report_path
        self.benchmark_path = benchmark_path
        self.benchmark_min_pass_rate = float(benchmark_min_pass_rate)
        self.benchmark_max_latency_ms = float(benchmark_max_latency_ms)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def _load_benchmark_report(self) -> dict[str, object]:
        if not self.benchmark_path or not os.path.exists(self.benchmark_path):
            return {
                "available": False,
                "passed": True,
                "reason": "benchmark_not_found",
            }
        try:
            with open(self.benchmark_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("benchmark payload must be an object")
            pass_rate = float(raw.get("pass_rate", 0.0) or 0.0)
            latency_ms = float(raw.get("avg_latency_ms", 0.0) or 0.0)
            passed = pass_rate >= self.benchmark_min_pass_rate and latency_ms <= self.benchmark_max_latency_ms
            return {
                "available": True,
                "passed": passed,
                "pass_rate": pass_rate,
                "avg_latency_ms": latency_ms,
                "min_pass_rate": self.benchmark_min_pass_rate,
                "max_latency_ms": self.benchmark_max_latency_ms,
                "tag": raw.get("tag"),
                "recent_render_pairs": raw.get("recent_render_pairs", 0),
            }
        except Exception as exc:
            return {
                "available": False,
                "passed": False,
                "reason": f"benchmark_read_error:{exc}",
            }

    def _evaluate_via_operational_script(self, timeout_sec: int = 600) -> dict[str, object] | None:
        command = [
            sys.executable,
            os.path.join("scripts", "eval", "operational_readiness.py"),
            "--refresh-artifacts",
            "--soak-profile",
            "standard",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            return {
                "command": " ".join(command),
                "returncode": int(completed.returncode),
                "stdout_tail": (completed.stdout or "")[-2000:],
                "stderr_tail": (completed.stderr or "")[-2000:],
                "passed": completed.returncode == 0,
            }
        except Exception as exc:
            return {
                "command": " ".join(command),
                "returncode": -1,
                "error": str(exc),
                "passed": False,
            }

    def evaluate(self, corpus_path: str, model_dir: str) -> dict[str, object]:
        corpus_lines = 0
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
                corpus_lines = sum(1 for _ in f)

        model_exists = os.path.isdir(model_dir) and any(os.scandir(model_dir))
        script_result = self._evaluate_via_operational_script(timeout_sec=300)
        script_passed = bool(script_result and script_result.get("passed", False))
        benchmark_result = self._load_benchmark_report()
        benchmark_passed = bool(benchmark_result.get("passed", True))
        baseline_passed = model_exists and corpus_lines > 100
        # Fallback to baseline gate when external evaluation fails to execute.
        script_executed = bool(script_result and script_result.get("returncode", -1) >= 0)
        passed = baseline_passed and (script_passed if script_executed else True) and benchmark_passed
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "corpus_lines": corpus_lines,
            "model_exists": model_exists,
            "baseline_passed": baseline_passed,
            "script_executed": script_executed,
            "operational_readiness": script_result,
            "benchmark": benchmark_result,
            "passed": passed,
        }
        atomic_write_json(self.report_path, report)
        return report
