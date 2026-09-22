#!/usr/bin/env python3
"""Audit frozen Backblaze result semantics without reading SMART rows or rerunning learners."""
from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import processed_data_path, workspace_path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_2013 = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_case_manifest_protocol_v1.json"))
PROTOCOL_2014 = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2014_case_manifest_protocol_v1.json"))
MANIFEST_2013 = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_case_manifest_v1.jsonl"))
MANIFEST_2014 = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2014_case_manifest_v1.jsonl"))
SCRIPTS = {
    "2013_materializer": ROOT / "scripts/eval/backblaze_2013_event_materialize.py",
    "2013_sparse_selectivity": ROOT / "scripts/eval/backblaze_2013_sparse_selectivity_diagnostic.py",
    "cross_year": ROOT / "scripts/eval/backblaze_cross_year_compact_development.py",
}
OUTPUT = Path(workspace_path("evaluation", "backblaze_prediction_time_audit_v1.json"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _has_endpoint_inclusive_window(tree: ast.AST) -> bool:
    """Find the row-date <= endpoint-date guard that admits terminal-day rows."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not any(
            isinstance(operator, (ast.LtE, ast.Gt)) for operator in node.ops
        ):
            continue
        source = ast.unparse(node)
        if "endpoint_date" in source and (
            "observed" in source or "row[date_index]" in source
        ):
            return True
    return False


def _cross_year_training_cutoff(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "training_histories"
                   for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for arg in node.value.args:
            if not isinstance(arg, ast.Call) or not isinstance(arg.func, ast.Name):
                continue
            if arg.func.id == "date" and len(arg.args) == 3:
                return "-".join(f"{value.value:02d}" for value in arg.args)
    raise ValueError("Could not identify cross-year training cutoff")


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Backblaze prediction-time audit already exists")
    protocols = {
        "2013": json.loads(PROTOCOL_2013.read_text()),
        "2014": json.loads(PROTOCOL_2014.read_text()),
    }
    manifests = {"2013": _rows(MANIFEST_2013), "2014": _rows(MANIFEST_2014)}
    trees = {name: ast.parse(path.read_text()) for name, path in SCRIPTS.items()}
    cutoff = _cross_year_training_cutoff(trees["cross_year"])
    training_2013 = [row for row in manifests["2013"] if row["split"] == "training"]
    after_cutoff = [row for row in training_2013 if row["endpoint_date"] > cutoff]
    positive_counts = {
        year: dict(Counter(row["split"] for row in rows if row["label"] == 1))
        for year, rows in manifests.items()
    }
    negative_endpoint_rules = {
        year: protocol["case_definition"]["negative_endpoint"]
        for year, protocol in protocols.items()
    }
    inclusive = {
        name: _has_endpoint_inclusive_window(tree)
        for name, tree in trees.items()
    }
    checks = {
        "terminal_day_included_in_all_experiment_windows": all(inclusive.values()),
        "negative_endpoints_defined_by_last_observation": all(
            "last observed day" in rule for rule in negative_endpoint_rules.values()),
        "cross_year_training_cutoff_precedes_manifest_end": bool(after_cutoff),
        "heldout_smart_values_not_accessed_by_this_audit": True,
    }
    result = {
        "schema": "sara-backblaze-prediction-time-audit-v1",
        "source_sha256": {name: _digest(path) for name, path in SCRIPTS.items()},
        "manifest_sha256": {"2013": _digest(MANIFEST_2013), "2014": _digest(MANIFEST_2014)},
        "endpoint_inclusive_window_by_script": inclusive,
        "positive_case_counts_by_split": positive_counts,
        "negative_endpoint_rule_by_year": negative_endpoint_rules,
        "cross_year_training_cutoff": cutoff,
        "cross_year_training_manifest_end": protocols["2013"]["split_boundaries"]["training_end"],
        "cross_year_training_cases_after_cutoff": len(after_cutoff),
        "cross_year_training_cases_after_cutoff_by_month": dict(Counter(
            row["endpoint_date"][:7] for row in after_cutoff)),
        "checks": checks,
        "prospective_prediction_claim_allowed": False,
        "cross_year_complete_training_claim_allowed": False,
        "prior_result_files_modified": False,
        "candidate_rerun": False,
        "smart_values_read": False,
        "heldout_consumed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"checks": checks,
                      "cross_year_training_cases_after_cutoff": len(after_cutoff),
                      "prospective_prediction_claim_allowed": False}, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
