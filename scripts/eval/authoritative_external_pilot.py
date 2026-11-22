"""Run a bounded, default-off pilot against a configured V2 HTTPS publisher."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.authoritative_evidence import (
    AuthoritativeEvidenceConfig, AuthoritativeEvidenceRuntime,
    validate_authoritative_evidence_config,
)
from sara_engine.memory.evidence_http import EvidenceHTTPClient
from sara_engine.memory.verified_chat_router import route_verified_chat
from sara_engine.utils.project_paths import (
    ensure_allowed_output_path, ensure_parent_directory, workspace_path,
)


CONFIG_SCHEMA = "sara-authoritative-external-pilot-config-v1"
REPORT_SCHEMA = "sara-authoritative-external-pilot-report-v1"
CONFIG_KEYS = {
    "schema", "endpoint", "publisher_id", "scope", "language", "aliases",
    "markers", "ca_file", "timeout_seconds", "max_bytes",
    "max_snapshot_lifetime_seconds", "max_future_skew_seconds",
    "max_fetch_seconds", "sequence_state_path", "questions",
}


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate pilot configuration key")
        value[key] = item
    return value


def load_configuration(path):
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("Pilot configuration exceeds 64 KiB")
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or set(payload) != CONFIG_KEYS:
        raise ValueError("Invalid pilot configuration fields")
    return payload


def parse_configuration(payload):
    if not isinstance(payload, dict) or set(payload) != CONFIG_KEYS:
        raise ValueError("Invalid pilot configuration fields")
    if payload["schema"] != CONFIG_SCHEMA:
        raise ValueError("Invalid pilot configuration schema")
    aliases = payload["aliases"]
    markers = payload["markers"]
    questions = payload["questions"]
    if (
        not isinstance(aliases, list) or len(aliases) > 64
        or any(not isinstance(row, list) or len(row) != 2 for row in aliases)
        or not isinstance(markers, list) or len(markers) > 16
        or not isinstance(questions, list) or not 1 <= len(questions) <= 12
        or any(not isinstance(value, str) or not 0 < len(value) <= 256 for value in questions)
        or payload["sequence_state_path"] is None
    ):
        raise ValueError("Invalid bounded pilot catalog")
    config = AuthoritativeEvidenceConfig(
        endpoint=payload["endpoint"], publisher_id=payload["publisher_id"],
        scope=payload["scope"], language=payload["language"],
        aliases=tuple(tuple(row) for row in aliases), markers=tuple(markers),
        ca_file=payload["ca_file"], timeout_seconds=payload["timeout_seconds"],
        max_bytes=payload["max_bytes"],
        max_snapshot_lifetime_seconds=payload["max_snapshot_lifetime_seconds"],
        max_future_skew_seconds=payload["max_future_skew_seconds"],
        max_fetch_seconds=payload["max_fetch_seconds"],
        sequence_state_path=payload["sequence_state_path"],
    )
    if any(
        route_verified_chat(
            question, language=config.language, aliases=config.aliases,
            markers=config.markers,
        ).decision != "verified"
        for question in questions
    ):
        raise ValueError("Pilot questions must be verified-scope questions")
    return config, tuple(questions)


def summarize_observations(refreshes, question_checks):
    """Summarize client observations without inferring independent acceptance."""
    count = len(refreshes)
    accepted = sum(
        row["decision"] in {"published", "publisher_not_modified"}
        for row in refreshes
    )
    metrics = {}
    for key in ("request_count", "response_bytes", "elapsed_seconds"):
        values = [row[key] for row in refreshes if row[key] is not None]
        metrics[key] = {
            "measured_cycles": len(values),
            "observed_total": sum(values) if values else None,
            "complete_total": sum(values) if len(values) == count and count else None,
        }
        if key == "elapsed_seconds":
            metrics[key]["observed_max"] = max(values) if values else None
    kinds = Counter()
    by_question = {}
    for row in question_checks:
        for index, kind in enumerate(row["response_kinds_in_question_order"]):
            kinds[kind] += 1
            by_question.setdefault(index, Counter())[kind] += 1
    checks = sum(kinds.values())
    return {
        "attempted_refreshes": count,
        "accepted_refreshes": accepted,
        "refresh_acceptance_rate": accepted / count if count else None,
        "question_checks": checks,
        "response_kind_counts": dict(sorted(kinds.items())),
        "verified_answer_rate": kinds["verified_answer"] / checks if checks else None,
        "response_counts_by_question": [
            {"question_index": index, "response_kind_counts": dict(sorted(counts.items()))}
            for index, counts in sorted(by_question.items())
        ],
        "metrics": metrics,
        "independent_availability_assessed": False,
        "authority_review_assessed": False,
        "revision_completeness_assessed": False,
        "complete_path_cost_assessed": False,
        "agreed_limits_assessed": False,
    }


def run_pilot(payload, *, cycles, client_factory=EvidenceHTTPClient,
              clock=None, performance_clock=None, interval_seconds=0):
    if type(cycles) is not int or not 1 <= cycles <= 32:
        raise ValueError("Pilot cycles must be between 1 and 32")
    if type(interval_seconds) is not int or not 0 <= interval_seconds <= 60:
        raise ValueError("Pilot interval must be between 0 and 60 seconds")
    config, questions = parse_configuration(payload)
    runtime = AuthoritativeEvidenceRuntime(
        config, clock=clock, client_factory=client_factory,
        performance_clock=performance_clock,
    )
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    refreshes = []
    question_checks = []
    decisions = Counter()
    for cycle in range(cycles):
        if cycle and interval_seconds:
            sleep(interval_seconds)
        sequence_before_refresh = runtime.accepted_sequence
        result = runtime.refresh()
        decisions[result.decision] += 1
        trace = runtime.get_last_refresh_trace()
        refreshes.append({
            "cycle": cycle,
            "accepted_sequence_before_refresh": sequence_before_refresh,
            **trace,
        })
        kinds = []
        for question in questions:
            runtime.chat(agent, question)
            kinds.append(agent.get_last_response_trace()["kind"])
        question_checks.append({
            "cycle": cycle, "question_count": len(questions),
            "response_kind_counts": dict(sorted(Counter(kinds).items())),
            "response_kinds_in_question_order": kinds,
        })
    return {
        "schema": REPORT_SCHEMA,
        "publisher_id": config.publisher_id,
        "scope": config.scope,
        "language": config.language,
        "cycle_count": cycles,
        "interval_seconds": interval_seconds,
        "decision_counts": dict(sorted(decisions.items())),
        "refreshes": refreshes,
        "question_checks": question_checks,
        "observation_summary": summarize_observations(refreshes, question_checks),
        "completed": True,
        "promotion_ready": False,
        "automatic_default_enabled": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cycles", type=int)
    mode.add_argument(
        "--check-config", action="store_true",
        help="Validate the configuration locally and print a summary.",
    )
    parser.add_argument(
        "--output",
        default=workspace_path("evaluation", "authoritative_external_pilot.json"),
    )
    parser.add_argument(
        "--interval-seconds", type=int, choices=range(61), metavar="0..60",
        help="Wait this many seconds between live cycles (default: 0).",
    )
    args = parser.parse_args(argv)
    if args.check_config and args.interval_seconds is not None:
        parser.error("--interval-seconds requires --cycles")
    return args


def validate_report_path(output, config_path, sequence_path):
    report = Path(ensure_allowed_output_path(output))
    sequence = Path(ensure_allowed_output_path(sequence_path))
    lock = Path(ensure_allowed_output_path(str(sequence) + ".lock"))
    for protected in (Path(config_path).resolve(), sequence, lock):
        if report == protected or (
            report.exists() and protected.exists() and report.samefile(protected)
        ):
            raise ValueError("Pilot report path overlaps configuration or sequence state")
    if report.exists() and not report.is_file():
        raise ValueError("Pilot report path must be a file")
    return report


def main(argv=None):
    args = parse_args(argv)
    payload = load_configuration(args.config)
    config, questions = parse_configuration(payload)
    output = validate_report_path(args.output, args.config, config.sequence_state_path)
    if args.check_config:
        validate_authoritative_evidence_config(config)
        EvidenceHTTPClient(
            config.endpoint, lambda value: value,
            timeout_seconds=config.timeout_seconds, max_bytes=config.max_bytes,
            ca_file=config.ca_file,
        )
        print(json.dumps({
            "configuration_valid": True,
            "question_count": len(questions),
        }))
        return
    ensure_parent_directory(str(output))
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=".pilot-report-", mode="w", encoding="utf-8",
        delete=False,
    ) as handle:
        pending_report = Path(handle.name)
        try:
            report = run_pilot(
                payload, cycles=args.cycles, interval_seconds=args.interval_seconds or 0,
            )
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.close()
            pending_report.replace(output)
        finally:
            pending_report.unlink(missing_ok=True)
    print(json.dumps({
        "schema": report["schema"], "cycle_count": report["cycle_count"],
        "decision_counts": report["decision_counts"],
        "completed": report["completed"],
        "promotion_ready": report["promotion_ready"],
        "automatic_default_enabled": report["automatic_default_enabled"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
