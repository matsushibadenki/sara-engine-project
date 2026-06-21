#!/usr/bin/env python3
"""Build a managed readiness report for the optional Rust sparse runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "rust_core_readiness.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "rust_core_readiness_summary.txt")
DEFAULT_BENCHMARK_REPORT_PATH = workspace_path("evaluation", "rust_core_benchmark.json")
EXPECTED_EXPORTS = (
    "calculate_sdr_overlap",
    "sparse_propagate_threshold",
    "build_direct_synapses",
    "batch_tokens_to_sdr",
    "apply_homeostatic_scaling",
    "SpikeEngine",
    "SpikeWTARouter",
    "LIFNetwork",
    "CausalSynapses",
    "ScalableSDRMemory",
    "RewardModulatedSTDP",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _has_non_english_code_comments(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("///"):
            if re.search(r"[ぁ-んァ-ン一-龯]", stripped):
                return True
    return False


def collect_version_alignment() -> Dict[str, Any]:
    cargo_text = _read_text(PROJECT_ROOT / "Cargo.toml")
    pyproject_text = _read_text(PROJECT_ROOT / "pyproject.toml")
    cargo_version = _extract_first(r'^version\s*=\s*"([^"]+)"', cargo_text)
    project_version = _extract_first(r'^version\s*=\s*"([^"]+)"', pyproject_text)
    maturin_features = _extract_first(r"^features\s*=\s*\[([^\]]*)\]", pyproject_text)
    return {
        "cargo_version": cargo_version,
        "project_version": project_version,
        "versions_match": bool(cargo_version and cargo_version == project_version),
        "cargo_extension_feature_declared": 'extension-module = ["pyo3/extension-module"]' in cargo_text,
        "maturin_uses_local_extension_feature": '"extension-module"' in maturin_features,
    }


def collect_export_contract() -> Dict[str, Any]:
    lib_text = _read_text(PROJECT_ROOT / "src" / "sara_engine" / "lib.rs")
    missing_wrappers = [
        name
        for name in EXPECTED_EXPORTS
        if f"wrap_pyfunction!({name}" not in lib_text and f"add_class::<{name}>" not in lib_text
    ]
    return {
        "expected_exports": list(EXPECTED_EXPORTS),
        "missing_from_pymodule_registration": missing_wrappers,
        "non_english_comments_present": _has_non_english_code_comments(lib_text),
        "batch_sdr_uses_rayon": "use rayon::prelude::*;" in lib_text
        and "batch_tokens.par_iter()" in lib_text,
    }


def collect_python_import_smoke() -> Dict[str, Any]:
    try:
        module = importlib.import_module("sara_engine.sara_rust_core")
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "missing_exports": list(EXPECTED_EXPORTS),
        }
    missing = [name for name in EXPECTED_EXPORTS if not hasattr(module, name)]
    return {
        "available": True,
        "module": getattr(module, "__name__", "sara_engine.sara_rust_core"),
        "missing_exports": missing,
    }


def collect_build_backend() -> Dict[str, Any]:
    maturin_path = shutil.which("maturin")
    python_maturin = run_command((sys.executable, "-m", "maturin", "--version"), PROJECT_ROOT)
    python_maturin_available = bool(python_maturin.get("passed"))
    return {
        "maturin_on_path": bool(maturin_path),
        "maturin_path": maturin_path or "",
        "python_maturin_available": python_maturin_available,
        "python_maturin_returncode": python_maturin.get("returncode"),
        "python_maturin_error_tail": python_maturin.get("stderr_tail", ""),
        "recommended_build_command": "python -m maturin develop --features extension-module",
    }


def run_command(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def build_report(run_cargo_test: bool = False) -> Dict[str, Any]:
    version_alignment = collect_version_alignment()
    export_contract = collect_export_contract()
    build_backend = collect_build_backend()
    python_smoke = collect_python_import_smoke()
    cargo_test = (
        run_command(("cargo", "test"), PROJECT_ROOT)
        if run_cargo_test
        else {"status": "not_run", "passed": None}
    )
    benchmark_report_exists = os.path.exists(DEFAULT_BENCHMARK_REPORT_PATH)
    checks = {
        "versions_match": version_alignment["versions_match"],
        "cargo_feature_split_ready": bool(
            version_alignment["cargo_extension_feature_declared"]
            and version_alignment["maturin_uses_local_extension_feature"]
        ),
        "pymodule_exports_registered": not export_contract["missing_from_pymodule_registration"],
        "rust_core_comments_english": not export_contract["non_english_comments_present"],
        "batch_sdr_parallelized": export_contract["batch_sdr_uses_rayon"],
        "python_extension_available": python_smoke["available"],
        "python_exports_complete": python_smoke["available"] and not python_smoke["missing_exports"],
        "maturin_build_backend_available": bool(
            build_backend["maturin_on_path"] or build_backend["python_maturin_available"]
        ),
        "benchmark_report_present": benchmark_report_exists,
        "cargo_test_passed": bool(cargo_test.get("passed")) if run_cargo_test else None,
    }
    required_for_source_readiness = (
        checks["versions_match"]
        and checks["cargo_feature_split_ready"]
        and checks["pymodule_exports_registered"]
        and checks["rust_core_comments_english"]
        and (checks["cargo_test_passed"] is not False)
    )
    required_for_built_extension = checks["python_extension_available"] and checks["python_exports_complete"]
    return {
        "schema": "sara-rust-core-readiness-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if required_for_source_readiness and required_for_built_extension else "needs_build_or_review",
        "source_readiness_passed": bool(required_for_source_readiness),
        "built_extension_readiness_passed": bool(required_for_built_extension),
        "checks": checks,
        "version_alignment": version_alignment,
        "export_contract": export_contract,
        "build_backend": build_backend,
        "python_import_smoke": python_smoke,
        "cargo_test": cargo_test,
        "benchmark_report": {
            "path": DEFAULT_BENCHMARK_REPORT_PATH,
            "present": benchmark_report_exists,
        },
        "policy_notes": [
            "Rust core remains CPU-first and sparse-event oriented.",
            "Python extension import may be unavailable until maturin builds the optional module.",
            "Direct cargo build of a PyO3 extension is not the release path on macOS; use maturin.",
            "Reports are written only under workspace/evaluation.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    checks = report.get("checks", {})
    lines = [
        f"Rust core readiness: {report.get('status')}",
        f"Source readiness: {report.get('source_readiness_passed')}",
        f"Built extension readiness: {report.get('built_extension_readiness_passed')}",
        f"Versions match: {checks.get('versions_match')}",
        f"Batch SDR parallelized: {checks.get('batch_sdr_parallelized')}",
        f"Cargo test passed: {checks.get('cargo_test_passed')}",
        f"Maturin build backend available: {checks.get('maturin_build_backend_available')}",
        f"Benchmark report present: {checks.get('benchmark_report_present')}",
        f"Python extension available: {checks.get('python_extension_available')}",
    ]
    missing = report.get("export_contract", {}).get("missing_from_pymodule_registration", [])
    if missing:
        lines.append("Missing registered exports: " + ", ".join(str(item) for item in missing))
    smoke_error = report.get("python_import_smoke", {}).get("error")
    if smoke_error:
        lines.append("Python smoke error: " + str(smoke_error))
    return "\n".join(lines) + "\n"


def write_report(report: Dict[str, Any], report_path: str, summary_path: str) -> Dict[str, str]:
    resolved_report = ensure_parent_directory(report_path)
    resolved_summary = ensure_parent_directory(summary_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    return {"report_path": resolved_report, "summary_path": resolved_summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--run-cargo-test", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(run_cargo_test=args.run_cargo_test)
    paths = write_report(report, args.report_path, args.summary_path)
    print(json.dumps({"status": report["status"], **paths}, ensure_ascii=False, indent=2))
    return 0 if report["source_readiness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
