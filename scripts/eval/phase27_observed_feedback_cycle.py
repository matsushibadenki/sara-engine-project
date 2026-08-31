#!/usr/bin/env python3
"""Replay a multi-step predictive cycle derived from observed source materials."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.edge.portable_decision_trace import (  # noqa: E402
    adapt_predictive_feedback,
    canonical_decision_json,
    decision_trace_digest,
)
from sara_engine.risa.structural_interpolation import (  # noqa: E402
    PredictiveStructuralFeedbackEngine,
    StructuralFeedbackSignal,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_SOURCE = processed_data_path("autobot", "phase27_observed_feedback_history.jsonl")
DEFAULT_OUTPUT = workspace_path("evaluation", "phase27_observed_feedback_cycle.json")


def load_rows(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(rows: Sequence[Mapping[str, Any]], rust_core: Any | None = None) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["time_segment"]))
    observed_changes = [
        float(previous["material_hash"] != current["material_hash"])
        for previous, current in zip(ordered, ordered[1:])
    ]
    engine = PredictiveStructuralFeedbackEngine(mismatch_threshold=0.25)
    actions: List[str] = []
    records: List[Dict[str, Any]] = []
    predicted = 0.5
    for index, (previous, current, observed) in enumerate(
        zip(ordered, ordered[1:], observed_changes)
    ):
        proposal = engine.propose(
            (
                StructuralFeedbackSignal(
                    predicting_concept="cpython-argparse-material-change",
                    source_node=str(previous["manifest_id"]),
                    target_node=str(current["manifest_id"]),
                    relation_type="next_revision_changes_material",
                    predicted_confidence=predicted,
                    observed_confidence=observed,
                    evidence_ids=(
                        str(previous["source_url"]),
                        str(current["source_url"]),
                        str(previous["material_hash"]),
                        str(current["material_hash"]),
                    ),
                    recent_actions=tuple(actions),
                ),
            )
        )[0]
        actions.append(proposal.edit_type)
        records.append(adapt_predictive_feedback(proposal, sequence=index))
        predicted = observed

    python_json = canonical_decision_json(records)
    python_digest = decision_trace_digest(records)
    canonical_fn = getattr(rust_core, "canonical_portable_decision_trace_json", None)
    digest_fn = getattr(rust_core, "portable_decision_trace_digest", None)
    rust_available = callable(canonical_fn) and callable(digest_fn)
    source = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    rust_json = str(canonical_fn(source)) if rust_available else ""
    rust_digest = str(digest_fn(source)) if rust_available else ""
    checks = {
        "four_ordered_observations": len(ordered) == 4
        and [int(row["time_segment"]) for row in ordered] == [0, 1, 2, 3],
        "same_logical_source": len({str(row["logical_source_ref"]) for row in ordered}) == 1,
        "independent_external_scope": all(
            row.get("evidence_scope") == "independent_external_observed_cycle"
            for row in ordered
        ),
        "observed_change_then_stability": observed_changes == [1.0, 1.0, 0.0],
        "multi_step_feedback": len(records) == 3,
        "expected_action_path": actions
        == ["strengthen_relation", "request_more_evidence", "cut_relation"],
        "no_false_contradiction": all(record["contradiction"] is False for record in records),
        "rust_extension_available": rust_available,
        "canonical_bytes_equivalent": rust_available and rust_json == python_json,
        "digest_equivalent": rust_available and rust_digest == python_digest,
    }
    return {
        "schema": "sara-phase27-observed-feedback-cycle-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "independent_evidence": True,
        "observed_change_path": observed_changes,
        "feedback_action_path": actions,
        "decision_trace_digest": python_digest,
        "rust_decision_trace_digest": rust_digest or None,
        "checks": checks,
        "metrics": {"source_revision_count": len(ordered), "feedback_step_count": len(records)},
        "claim_boundary": "Official source hashes support one three-transition material-change feedback cycle only; change is a byte-identity observation, not a semantic-quality or contradiction judgment.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", default=DEFAULT_SOURCE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        rust_core = importlib.import_module("sara_engine.sara_rust_core")
    except ImportError:
        rust_core = None
    report = build_report(load_rows(args.source_path), rust_core)
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
