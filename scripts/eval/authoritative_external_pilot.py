"""Run a bounded, default-off pilot against a configured V2 HTTPS publisher."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.authoritative_evidence import (
    AuthoritativeEvidenceConfig, AuthoritativeEvidenceRuntime,
)
from sara_engine.memory.evidence_http import EvidenceHTTPClient
from sara_engine.memory.verified_chat_router import route_verified_chat
from sara_engine.utils.project_paths import workspace_path, ensure_parent_directory


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


def run_pilot(payload, *, cycles, client_factory=EvidenceHTTPClient,
              clock=None, performance_clock=None):
    if type(cycles) is not int or not 1 <= cycles <= 32:
        raise ValueError("Pilot cycles must be between 1 and 32")
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
        result = runtime.refresh()
        decisions[result.decision] += 1
        trace = runtime.get_last_refresh_trace()
        refreshes.append({"cycle": cycle, **trace})
        kinds = Counter()
        for question in questions:
            runtime.chat(agent, question)
            kinds[agent.get_last_response_trace()["kind"]] += 1
        question_checks.append({
            "cycle": cycle, "question_count": len(questions),
            "response_kind_counts": dict(sorted(kinds.items())),
        })
    return {
        "schema": REPORT_SCHEMA,
        "publisher_id": config.publisher_id,
        "scope": config.scope,
        "language": config.language,
        "cycle_count": cycles,
        "decision_counts": dict(sorted(decisions.items())),
        "refreshes": refreshes,
        "question_checks": question_checks,
        "completed": True,
        "promotion_ready": False,
        "automatic_default_enabled": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument(
        "--output",
        default=workspace_path("evaluation", "authoritative_external_pilot.json"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_pilot(load_configuration(args.config), cycles=args.cycles)
    output = Path(ensure_parent_directory(args.output))
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "schema": report["schema"], "cycle_count": report["cycle_count"],
        "decision_counts": report["decision_counts"],
        "completed": report["completed"],
        "promotion_ready": report["promotion_ready"],
        "automatic_default_enabled": report["automatic_default_enabled"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
