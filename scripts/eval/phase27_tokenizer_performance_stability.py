#!/usr/bin/env python3
"""Aggregate fresh-process Phase 27 tokenizer measurements without run selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_output_directory,
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase27_tokenizer_conformance_cases.jsonl"
)
DEFAULT_TRIAL_DIR = workspace_path(
    "evaluation", "phase27_tokenizer_stability_trials"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase27_tokenizer_performance_stability.json"
)
BENCHMARK_SCRIPT = os.path.join(
    PROJECT_ROOT, "scripts", "eval", "phase27_tokenizer_acceleration_benchmark.py"
)
EXPECTED_SCHEMA = "sara-phase27-tokenizer-acceleration-benchmark-v2"
MIN_TRIALS = 5
MIN_SPEEDUP = 1.05


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    fixture_digest: str,
    required_trials: int = MIN_TRIALS,
    minimum_speedup: float = MIN_SPEEDUP,
) -> Dict[str, Any]:
    speeds: List[float] = []
    trial_rows: List[Dict[str, Any]] = []
    tokenizer_fingerprints = set()
    all_integrity = True
    for index, report in enumerate(reports):
        checks = report.get("checks", {})
        measurement = report.get("resource_measurement", {})
        speedup = measurement.get("rust_snapshot_median_speedup_vs_python")
        numeric_speedup = (
            float(speedup)
            if isinstance(speedup, (int, float)) and not isinstance(speedup, bool)
            else 0.0
        )
        speeds.append(numeric_speedup)
        tokenizer_fingerprints.add(str(report.get("tokenizer_fingerprint", "")))
        integrity = bool(
            report.get("schema") == EXPECTED_SCHEMA
            and report.get("passed") is True
            and report.get("observed_only") is True
            and report.get("production_path_changed") is False
            and report.get("rust_build_profile") == "release"
            and report.get("rust_snapshot_reference_equivalent") is True
            and isinstance(checks, Mapping)
            and checks.get("large_trace_snapshot_equivalent") is True
            and checks.get("rust_snapshot_downstream_replay_equivalent") is True
            and checks.get("snapshot_state_bounded") is True
            and checks.get("peak_rss_growth_bounded") is True
            and int(measurement.get("median_trace_count", 0)) == 300
            and int(measurement.get("median_repetitions", 0)) == 7
        )
        all_integrity = all_integrity and integrity
        trial_rows.append(
            {
                "trial_index": index,
                "integrity_passed": integrity,
                "speedup": numeric_speedup,
                "threshold_passed": numeric_speedup > minimum_speedup,
                "python_median_elapsed_ns": measurement.get("python_median_elapsed_ns"),
                "rust_snapshot_median_elapsed_ns": measurement.get(
                    "rust_snapshot_median_elapsed_ns"
                ),
                "peak_rss_delta_bytes": measurement.get("peak_rss_delta_bytes"),
            }
        )
    trial_count_ok = len(reports) >= required_trials
    fingerprints_frozen = len(tokenizer_fingerprints) == 1 and "" not in tokenizer_fingerprints
    worst_speedup = min(speeds) if speeds else 0.0
    median_speedup = float(statistics.median(speeds)) if speeds else 0.0
    every_trial_above_threshold = bool(speeds) and all(
        speedup > minimum_speedup for speedup in speeds
    )
    promotion_ready = bool(
        trial_count_ok
        and all_integrity
        and fingerprints_frozen
        and every_trial_above_threshold
    )
    checks = {
        "required_trial_count_present": trial_count_ok,
        "all_trial_integrity_passed": all_integrity and bool(reports),
        "tokenizer_fingerprint_frozen": fingerprints_frozen,
        "every_trial_above_threshold": every_trial_above_threshold,
        "production_path_unchanged": all(
            report.get("production_path_changed") is False for report in reports
        ),
    }
    return {
        "schema": "sara-phase27-tokenizer-performance-stability-v1",
        "passed": all(value is True for key, value in checks.items() if key != "every_trial_above_threshold"),
        "observed_only": True,
        "production_path_changed": False,
        "promotion_ready": promotion_ready,
        "selection_policy": "all_fresh_trials_retained_no_post_observation_exclusion",
        "fixture_digest": fixture_digest,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "thresholds": {
            "required_trials": required_trials,
            "minimum_speedup_exclusive": minimum_speedup,
            "required_trace_count": 300,
            "required_repetitions_per_trial": 7,
            "required_rust_profile": "release",
        },
        "checks": checks,
        "metrics": {
            "trial_count": len(reports),
            "median_speedup": median_speedup,
            "worst_speedup": worst_speedup,
            "best_speedup": max(speeds) if speeds else 0.0,
        },
        "trials": trial_rows,
        "claim_boundary": "Fresh-process performance stability for one frozen tokenizer fixture only; no general throughput, energy, or production-readiness claim.",
    }


def run_fresh_trials(
    *, fixture_path: str, trial_dir: str, trial_count: int
) -> List[Mapping[str, Any]]:
    output_dir = ensure_output_directory(trial_dir)
    reports: List[Mapping[str, Any]] = []
    for index in range(trial_count):
        output_path = os.path.join(output_dir, f"trial_{index:02d}.json")
        subprocess.run(
            [
                sys.executable,
                BENCHMARK_SCRIPT,
                "--fixture-path",
                fixture_path,
                "--output-path",
                output_path,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        reports.append(load_report(output_path))
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--trial-dir", default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--trial-count", type=int, default=MIN_TRIALS)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.trial_count < MIN_TRIALS:
        parser.error(f"--trial-count must be at least {MIN_TRIALS}")
    reports = run_fresh_trials(
        fixture_path=args.fixture_path,
        trial_dir=args.trial_dir,
        trial_count=args.trial_count,
    )
    report = aggregate_reports(
        reports,
        fixture_digest=_file_digest(args.fixture_path),
        required_trials=args.trial_count,
    )
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
