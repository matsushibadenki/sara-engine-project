#!/usr/bin/env python3
"""Build the Phase 34 semantic delayed-recall fixture and immutable draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_human_review import (  # noqa: E402
    canonical_digest,
    validate_ledger,
    validate_request,
)
from sara_engine.evaluation.phase34_semantic_preregistration import (  # noqa: E402
    BUDGETS,
    CASE_COUNT,
    CASE_FAMILIES,
    CLAIM_BOUNDARIES,
    COMPARISON_PACKET_FINGERPRINT,
    EXECUTION_POLICY,
    EXPERIMENT_ID,
    HORIZONS,
    LANGUAGES,
    PARENT_PROTOCOL_FINGERPRINT,
    PARENT_REPORT_FINGERPRINT,
    REPLICATE_SEEDS,
    REVIEW_GATE_REPORT_FINGERPRINT,
    REVIEW_LEDGER_FINGERPRINT,
    REVIEW_REQUEST_FINGERPRINT,
    REVIEW_SUPPORT_SNAPSHOT_FINGERPRINT,
    SCHEMA,
    TARGET_IDS,
    THRESHOLDS,
    build_registered_manifest,
)
from sara_engine.evaluation.phase34_factorial_preregistration import ARMS  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


CASE_SCHEMA = "sara-phase34-semantic-delayed-recall-case-v1"
DEFAULT_REQUEST = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_request.json"
)
DEFAULT_LEDGER = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_decisions.json"
)
DEFAULT_GATE = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_gate.json"
)
DEFAULT_PACKET = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_review_comparison_packet.json"
)
DEFAULT_PARENT_PREREGISTRATION = workspace_path(
    "evaluation",
    "phase34_memory_cache_factorial_independent_adapter_v2_preregistration.json",
)
DEFAULT_PARENT_REPORT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_adapter_v2_benchmark.json"
)
DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase34_semantic_delayed_recall_cases.jsonl"
)
DEFAULT_DRAFT = workspace_path(
    "evaluation", "phase34_semantic_delayed_recall_preregistration_draft.json"
)
DEFAULT_ENVIRONMENT = workspace_path(
    "evaluation", "phase34_semantic_delayed_recall_environment.json"
)


PROBES: Dict[str, Dict[str, Any]] = {
    "arch-migration-ietf-001": {
        "proposition_id": "http_is_stateless_application_level_protocol",
        "positive": {
            "en": "At what protocol layer does HTTP operate, and is it stateful?",
            "ja": "HTTPはどのプロトコル層で動作し、状態を保持するプロトコルですか。",
            "zh-Hans": "HTTP在哪个协议层运行，它是否是有状态协议？",
        },
        "decoy": {
            "en": "Which transport-encryption algorithm does RFC 9110 require HTTP to use?",
            "ja": "RFC 9110はHTTPにどのトランスポート暗号化アルゴリズムを要求していますか。",
            "zh-Hans": "RFC 9110要求HTTP使用哪种传输加密算法？",
        },
    },
    "arch-migration-ietf-002": {
        "proposition_id": "http_messages_are_requests_or_responses",
        "positive": {
            "en": "What are the two message roles used by HTTP's uniform resource interface?",
            "ja": "HTTPの統一されたリソースインターフェースで使われる二つのメッセージ役割は何ですか。",
            "zh-Hans": "HTTP统一资源接口使用的两种消息角色是什么？",
        },
        "decoy": {
            "en": "Which database schema must every HTTP resource use internally?",
            "ja": "すべてのHTTPリソースが内部で使う必要のあるデータベーススキーマは何ですか。",
            "zh-Hans": "每个HTTP资源在内部必须使用哪种数据库模式？",
        },
    },
    "arch-migration-ietf-003": {
        "proposition_id": "bcp14_keywords_apply_only_in_all_capitals",
        "positive": {
            "en": "When do words such as MUST and SHOULD receive their BCP 14 meaning?",
            "ja": "MUSTやSHOULDなどの語がBCP 14の意味を持つのはどのような場合ですか。",
            "zh-Hans": "MUST和SHOULD等词在什么情况下具有BCP 14规定的含义？",
        },
        "decoy": {
            "en": "Which numeric HTTP status code is assigned to every violated MUST?",
            "ja": "違反されたすべてのMUSTには、どのHTTP数値ステータスコードが割り当てられますか。",
            "zh-Hans": "每个被违反的MUST都对应哪个数字HTTP状态码？",
        },
    },
    "arch-migration-python-001": {
        "proposition_id": "argparse_parses_sys_argv_and_generates_help",
        "positive": {
            "en": "Which module parses declared command-line arguments from sys.argv and can generate help and usage messages?",
            "ja": "宣言されたコマンドライン引数をsys.argvから解析し、ヘルプと使用法を生成できるモジュールは何ですか。",
            "zh-Hans": "哪个模块可以从sys.argv解析已声明的命令行参数并生成帮助和用法信息？",
        },
        "decoy": {
            "en": "How many worker threads does argparse create when parsing invalid arguments?",
            "ja": "argparseは不正な引数を解析するときに何本のワーカースレッドを作成しますか。",
            "zh-Hans": "argparse在解析无效参数时会创建多少个工作线程？",
        },
    },
    "arch-migration-python-002": {
        "proposition_id": "parse_args_returns_values_in_argparse_namespace",
        "positive": {
            "en": "After add_argument attaches argument specifications, where does parse_args place the extracted values?",
            "ja": "add_argumentが引数仕様を追加した後、parse_argsは抽出した値をどこに格納しますか。",
            "zh-Hans": "add_argument添加参数规范后，parse_args会把提取出的值存放在哪里？",
        },
        "decoy": {
            "en": "Which database table does add_argument use to persist parsed values?",
            "ja": "add_argumentは解析した値を永続化するために、どのデータベース表を使いますか。",
            "zh-Hans": "add_argument使用哪个数据库表来持久化解析后的值？",
        },
    },
    "arch-migration-python-003": {
        "proposition_id": "pure_paths_no_io_concrete_paths_add_io",
        "positive": {
            "en": "Which pathlib path category performs no I/O, and which category adds I/O operations?",
            "ja": "pathlibでは、どのパス分類がI/Oを行わず、どの分類がI/O操作を追加しますか。",
            "zh-Hans": "在pathlib中，哪类路径不执行I/O，哪类路径增加了I/O操作？",
        },
        "decoy": {
            "en": "Which network protocol does pathlib use to synchronize pure paths?",
            "ja": "pathlibはpure pathを同期するために、どのネットワークプロトコルを使いますか。",
            "zh-Hans": "pathlib使用哪种网络协议来同步pure path？",
        },
    },
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def build_cases(request: Mapping[str, Any]) -> List[Dict[str, Any]]:
    validation = validate_request(request)
    if not validation["valid"] or validation["request_fingerprint"] != REVIEW_REQUEST_FINGERPRINT:
        raise ValueError("semantic workload requires the frozen human-review request")
    targets = {str(item["record_id"]): item for item in request["targets"]}
    if set(targets) != set(TARGET_IDS) or set(PROBES) != set(TARGET_IDS):
        raise ValueError("semantic target set does not match the frozen six-source review")
    cases: List[Dict[str, Any]] = []
    for record_id in TARGET_IDS:
        target = targets[record_id]
        probe = PROBES[record_id]
        proposition_id = str(probe["proposition_id"])
        for language in LANGUAGES:
            for horizon in HORIZONS:
                for family in CASE_FAMILIES:
                    if family == "semantic_paraphrase_recall":
                        query = probe["positive"][language]
                        control_mode = "none"
                        expected_decision = "retrieve_original"
                        expected_proposition = proposition_id
                        independent = True
                    elif family == "lexical_overlap_abstention":
                        query = probe["decoy"][language]
                        control_mode = "none"
                        expected_decision = "abstain_unsupported"
                        expected_proposition = None
                        independent = False
                    elif family == "revision_replacement":
                        query = probe["positive"][language]
                        control_mode = "verified_later_revision"
                        expected_decision = "retrieve_revision"
                        expected_proposition = f"revision:{proposition_id}"
                        independent = False
                    elif family == "contradiction_abstention":
                        query = probe["positive"][language]
                        control_mode = "unresolved_equal_source_contradiction"
                        expected_decision = "abstain_contradiction"
                        expected_proposition = None
                        independent = False
                    else:
                        query = probe["positive"][language]
                        control_mode = "target_evidence_omitted"
                        expected_decision = "abstain_missing"
                        expected_proposition = None
                        independent = False
                    cases.append(
                        {
                            "schema": CASE_SCHEMA,
                            "case_id": f"p34-semantic:{record_id}:{language}:h{horizon}:{family}",
                            "record_id": record_id,
                            "language": language,
                            "horizon": horizon,
                            "family": family,
                            "query_text": query,
                            "control_mode": control_mode,
                            "expected_decision": expected_decision,
                            "expected_proposition_id": expected_proposition,
                            "source_hash": target["source_hash"],
                            "source_ref": target["source_ref"],
                            "source_revision": target["source_revision"],
                            "independent_semantic_evidence": independent,
                            "synthetic_control": not independent,
                            "observed_only": True,
                            "durable_mutation_allowed": False,
                        }
                    )
    validate_fixture(cases)
    return cases


def validate_fixture(rows: Sequence[Mapping[str, Any]]) -> None:
    errors: List[str] = []
    if len(rows) != CASE_COUNT:
        errors.append("semantic_case_count_mismatch")
    if [row.get("case_id") for row in rows] != list(dict.fromkeys(row.get("case_id") for row in rows)):
        errors.append("semantic_case_ids_must_be_unique")
    expected_distribution = Counter(
        (target, language, horizon, family)
        for target in TARGET_IDS
        for language in LANGUAGES
        for horizon in HORIZONS
        for family in CASE_FAMILIES
    )
    actual_distribution = Counter(
        (row.get("record_id"), row.get("language"), row.get("horizon"), row.get("family"))
        for row in rows
    )
    if actual_distribution != expected_distribution:
        errors.append("semantic_factorial_distribution_mismatch")
    for row in rows:
        if row.get("schema") != CASE_SCHEMA:
            errors.append("semantic_case_schema_mismatch")
        if not str(row.get("query_text", "")).strip():
            errors.append("semantic_query_missing")
        if row.get("observed_only") is not True or row.get("durable_mutation_allowed") is not False:
            errors.append("semantic_case_policy_mismatch")
        independent = row.get("independent_semantic_evidence")
        if independent is not (row.get("family") == "semantic_paraphrase_recall"):
            errors.append("semantic_independent_scope_mismatch")
        if row.get("synthetic_control") is independent:
            errors.append("semantic_control_scope_mismatch")
        if not isinstance(row.get("source_hash"), str) or len(row["source_hash"]) != 64:
            errors.append("semantic_source_hash_invalid")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))


def environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-semantic-delayed-recall-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
    }


def build_draft(
    rows: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    ledger: Mapping[str, Any],
    gate: Mapping[str, Any],
    packet: Mapping[str, Any],
    parent_preregistration: Mapping[str, Any],
    parent_report: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_fixture(rows)
    request_validation = validate_request(request)
    ledger_validation = validate_ledger(request, ledger)
    if (
        not request_validation["valid"]
        or request_validation["request_fingerprint"] != REVIEW_REQUEST_FINGERPRINT
        or not ledger_validation["valid"]
        or canonical_digest(dict(ledger)) != REVIEW_LEDGER_FINGERPRINT
    ):
        raise ValueError("semantic workload human-review evidence mismatch")
    if (
        gate.get("report_fingerprint") != REVIEW_GATE_REPORT_FINGERPRINT
        or gate.get("review_gate_passed") is not True
        or gate.get("semantic_delayed_recall_preregistration_ready") is not True
        or gate.get("promotion_ready") is not False
    ):
        raise ValueError("semantic workload human-review gate is not open")
    if (
        packet.get("packet_fingerprint") != COMPARISON_PACKET_FINGERPRINT
        or packet.get("source_snapshot_fingerprint") != REVIEW_SUPPORT_SNAPSHOT_FINGERPRINT
        or packet.get("request_fingerprint") != REVIEW_REQUEST_FINGERPRINT
    ):
        raise ValueError("semantic workload comparison packet mismatch")
    if (
        parent_preregistration.get("protocol_fingerprint") != PARENT_PROTOCOL_FINGERPRINT
        or parent_report.get("protocol_fingerprint") != PARENT_PROTOCOL_FINGERPRINT
        or _digest(dict(parent_report)) != PARENT_REPORT_FINGERPRINT
        or parent_report.get("execution_passed") is not True
        or parent_report.get("identity_gate_passed") is not True
        or parent_report.get("promotion_ready") is not False
    ):
        raise ValueError("semantic workload parent evidence mismatch")
    draft = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_protocol_fingerprint": PARENT_PROTOCOL_FINGERPRINT,
        "parent_report_fingerprint": PARENT_REPORT_FINGERPRINT,
        "review_request_fingerprint": REVIEW_REQUEST_FINGERPRINT,
        "review_ledger_fingerprint": REVIEW_LEDGER_FINGERPRINT,
        "review_gate_report_fingerprint": REVIEW_GATE_REPORT_FINGERPRINT,
        "comparison_packet_fingerprint": COMPARISON_PACKET_FINGERPRINT,
        "review_support_snapshot_fingerprint": REVIEW_SUPPORT_SNAPSHOT_FINGERPRINT,
        "registered_before_execution": True,
        "registered_before_semantic_adapter_implementation": True,
        "fixture_fingerprint": _digest(list(rows)),
        "environment_fingerprint": _digest(dict(environment)),
        "source_target_ids": list(TARGET_IDS),
        "languages": list(LANGUAGES),
        "horizons": list(HORIZONS),
        "case_families": list(CASE_FAMILIES),
        "case_count": CASE_COUNT,
        "arms": list(ARMS),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "replicates_per_condition": len(REPLICATE_SEEDS),
        "budgets": BUDGETS,
        "thresholds": THRESHOLDS,
        "claim_boundaries": CLAIM_BOUNDARIES,
        "execution_policy": EXECUTION_POLICY,
        "evaluation_contract": {
            "candidate_visible_fields": [
                "case_id", "record_id", "language", "horizon", "family",
                "query_text", "control_mode",
            ],
            "evaluator_only_fields": [
                "expected_decision", "expected_proposition_id", "source_hash",
                "source_ref", "source_revision", "independent_semantic_evidence",
                "synthetic_control",
            ],
            "macro_average_axes": ["record_id", "language", "horizon"],
            "exact_identity_score_is_semantic_score": False,
            "token_overlap_is_semantic_score": False,
            "source_trace_required_for_non_abstaining_answer": True,
        },
    }
    build_registered_manifest(draft, managed_path=True)
    return draft


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-path", default=DEFAULT_REQUEST)
    parser.add_argument("--ledger-path", default=DEFAULT_LEDGER)
    parser.add_argument("--gate-path", default=DEFAULT_GATE)
    parser.add_argument("--packet-path", default=DEFAULT_PACKET)
    parser.add_argument("--parent-preregistration-path", default=DEFAULT_PARENT_PREREGISTRATION)
    parser.add_argument("--parent-report-path", default=DEFAULT_PARENT_REPORT)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--draft-path", default=DEFAULT_DRAFT)
    parser.add_argument("--environment-path", default=DEFAULT_ENVIRONMENT)
    args = parser.parse_args(argv)
    request = _read_json(args.request_path)
    rows = build_cases(request)
    environment = environment_descriptor()
    draft = build_draft(
        rows,
        request,
        _read_json(args.ledger_path),
        _read_json(args.gate_path),
        _read_json(args.packet_path),
        _read_json(args.parent_preregistration_path),
        _read_json(args.parent_report_path),
        environment,
    )
    fixture_path = ensure_parent_directory(args.fixture_path)
    with open(fixture_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    for path, value in ((args.environment_path, environment), (args.draft_path, draft)):
        with open(ensure_parent_directory(path), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(
        json.dumps(
            {
                "schema": "sara-phase34-semantic-delayed-recall-draft-receipt-v1",
                "case_count": len(rows),
                "condition_count": len(rows) * len(ARMS) * len(REPLICATE_SEEDS),
                "fixture_fingerprint": draft["fixture_fingerprint"],
                "environment_fingerprint": draft["environment_fingerprint"],
                "fixture_path": os.path.realpath(args.fixture_path),
                "draft_path": os.path.realpath(args.draft_path),
                "environment_path": os.path.realpath(args.environment_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
