#!/usr/bin/env python3
"""Record one human excerpt review and evaluate the Phase 34 review gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_human_review import (  # noqa: E402
    build_empty_ledger,
    evaluate_review_gate,
    record_decision,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    workspace_path,
)


DEFAULT_REQUEST = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_request.json"
)
DEFAULT_LEDGER = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_decisions.json"
)
DEFAULT_REPORT = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_gate.json"
)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def _write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    resolved = ensure_parent_directory(path)
    parent = os.path.dirname(resolved)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".phase34-review-", suffix=".json", dir=parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, resolved)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _distortion_value(value: str) -> Optional[bool]:
    if value == "found":
        return True
    if value == "not-found":
        return False
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-path", default=DEFAULT_REQUEST)
    parser.add_argument("--ledger-path", default=DEFAULT_LEDGER)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--record-id")
    parser.add_argument("--authoritative-section-locator")
    parser.add_argument("--authoritative-text-hash")
    parser.add_argument(
        "--alignment-decision",
        choices=("aligned", "misaligned", "unresolved"),
    )
    parser.add_argument(
        "--semantic-distortion",
        choices=("found", "not-found", "unresolved"),
    )
    parser.add_argument("--reviewer")
    parser.add_argument("--reviewed-at")
    parser.add_argument("--notes", default="")
    parser.add_argument("--attest-human-review", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)

    try:
        request = _read_json(args.request_path)
        if os.path.exists(args.ledger_path):
            ledger = _read_json(args.ledger_path)
        else:
            ledger = build_empty_ledger(request)

        if args.record_id:
            required = {
                "--authoritative-section-locator": args.authoritative_section_locator,
                "--authoritative-text-hash": args.authoritative_text_hash,
                "--alignment-decision": args.alignment_decision,
                "--semantic-distortion": args.semantic_distortion,
                "--reviewer": args.reviewer,
                "--reviewed-at": args.reviewed_at,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "recording a review requires: " + ", ".join(sorted(missing))
                )
            ledger = record_decision(
                request,
                ledger,
                record_id=args.record_id,
                authoritative_section_locator=args.authoritative_section_locator,
                authoritative_text_hash=args.authoritative_text_hash,
                alignment_decision=args.alignment_decision,
                semantic_omission_or_distortion_found=_distortion_value(
                    args.semantic_distortion
                ),
                reviewer=args.reviewer,
                reviewed_at=args.reviewed_at,
                notes=args.notes,
                human_attestation=args.attest_human_review,
                replace_existing=args.replace_existing,
            )
            _write_json_atomic(args.ledger_path, ledger)
        elif any(
            value is not None
            for value in (
                args.authoritative_section_locator,
                args.authoritative_text_hash,
                args.alignment_decision,
                args.semantic_distortion,
                args.reviewer,
                args.reviewed_at,
            )
        ) or args.attest_human_review or args.replace_existing:
            raise ValueError("review fields require --record-id")

        report = evaluate_review_gate(request, ledger)
        _write_json_atomic(args.report_path, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "schema": report["schema"],
                "review_complete": report["review_complete"],
                "review_gate_passed": report["review_gate_passed"],
                "decision_count": report["decision_count"],
                "pending_count": report["pending_count"],
                "next_action": report["next_action"],
                "ledger_path": os.path.realpath(args.ledger_path),
                "report_path": os.path.realpath(args.report_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
