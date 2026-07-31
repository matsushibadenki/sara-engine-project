# {
#     "//": "ディレクトリパス: scripts/sara_cli.py",
#     "//": "ファイルの日本語タイトル: SARA統合コマンドラインインターフェース",
#     "//": "ファイルの目的や内容: データ収集、DB管理、そして【自己組織化学習】と【蒸留学習】の切り替えを一元管理する統合CLI。"
# }

import argparse
import sys
import os
import json
import shutil
import subprocess

# scriptsディレクトリ自体をシステムパスに追加
scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(scripts_dir, ".."))
src_dir = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, scripts_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# Keep CLI-launched evaluator subprocesses on the same source tree even when
# the package has not been installed into the active Python environment.
_pythonpath_parts = [src_dir, project_root]
if os.environ.get("PYTHONPATH"):
    _pythonpath_parts.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(_pythonpath_parts)

from sara_engine.utils.project_paths import (
    ensure_parent_directory,
    interim_data_path,
    model_path,
    processed_data_path,
    workspace_path,
)
from scripts.utils.manage_db import SaraCorpusDB


def default_replay_output_path() -> str:
    return workspace_path("replay", "chat_replay_tokens.jsonl")


def default_memory_health_report_path() -> str:
    return workspace_path("reports", "memory_health_report.json")


def default_upgraded_model_path() -> str:
    return model_path("upgraded", "distilled_sara_llm_upgraded.msgpack")


def default_upgrade_report_path() -> str:
    return workspace_path("reports", "memory_upgrade_report.json")


def default_fixed_model_path() -> str:
    return model_path("repaired", "memory_fixed.msgpack")


def default_fix_report_path() -> str:
    return workspace_path("reports", "memory_fix_report.json")


def build_replay_data(*args, **kwargs):
    from scripts.utils.build_replay_data import build_replay_data as implementation

    return implementation(*args, **kwargs)


def inspect_inference_memory(*args, **kwargs):
    from scripts.utils.memory_health import inspect_inference_memory as implementation

    return implementation(*args, **kwargs)


def upgrade_inference_memory(*args, **kwargs):
    from scripts.utils.upgrade_memory import upgrade_inference_memory as implementation

    return implementation(*args, **kwargs)


def fix_inference_memory(*args, **kwargs):
    from scripts.utils.fix_memory import fix_inference_memory as implementation

    return implementation(*args, **kwargs)


def prune_model_memory(*args, **kwargs):
    from scripts.utils.prune_memory import prune_model_memory as implementation

    return implementation(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="SARA Engine 統合管理CLI - Data & Learning Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # --- 1. データ登録・管理 (Database) ---
    parser_db_import = subparsers.add_parser("db-import", help="テキスト(.txt)や対話データ(.jsonl)をDBに取り込みます。")
    parser_db_import.add_argument("file", help="取り込むファイルパス")
    parser_db_import.add_argument("--category", default=None, help="学習素材カテゴリ。例: document, dialogue, research, manual, code")
    parser_db_import.add_argument("--lang", default="ja", help="素材の言語コード。例: ja, en")
    parser_db_import.add_argument("--source-version", default="", help="素材ソースの版や日付ラベル。")
    parser_db_import.add_argument("--quality-score", type=float, default=1.0, help="0.0-1.0 の品質スコア。")
    parser_db_import.add_argument("--inactive", action="store_true", help="取り込み後に inactive 素材として保持します。")
    parser_db_import.add_argument("--report", default=None, help="Optional JSON import report path under workspace/.")

    parser_db_status = subparsers.add_parser("db-status", help="現在のコーパスDBの登録件数を表示します。")
    parser_db_status.add_argument("--format", choices=["text", "json"], default="text", help="出力形式。")

    parser_db_list = subparsers.add_parser("db-list", help="学習候補素材を preview 付きで一覧表示します。")
    parser_db_list.add_argument("--category", default=None, help="指定カテゴリだけを表示します。")
    parser_db_list.add_argument("--source", default=None, help="指定 source 名の素材だけを表示します。")
    parser_db_list.add_argument("--min-quality-score", type=float, default=0.0, help="この品質スコア以上の素材だけを表示します。")
    parser_db_list.add_argument("--show-inactive", action="store_true", help="inactive 素材も一覧に含めます。")
    parser_db_list.add_argument("--limit", type=int, default=20, help="表示する最大件数。")
    parser_db_list.add_argument("--format", choices=["text", "json"], default="text", help="出力形式。")
    
    parser_db_export = subparsers.add_parser("db-export", help="DBから自己組織化用(TXT)と蒸留用(JSONL)にデータを一括出力します。")
    parser_db_export.add_argument("--category", default=None, help="指定カテゴリだけを export します。")
    parser_db_export.add_argument("--source", default=None, help="指定 source 名の素材だけを export します。")
    parser_db_export.add_argument("--min-quality-score", type=float, default=0.0, help="この品質スコア以上の素材だけを export します。")
    parser_db_export.add_argument("--show-inactive", action="store_true", help="inactive 素材も export / dry-run 対象に含めます。")
    parser_db_export.add_argument("--dry-run", action="store_true", help="実際には export せず、対象件数だけを表示します。")
    parser_db_export.add_argument("--report", default=None, help="Optional JSON export report path under workspace/.")

    parser_db_activate = subparsers.add_parser("db-activate", help="条件に一致する素材を active に切り替えます。")
    parser_db_activate.add_argument("--category", default=None, help="指定カテゴリだけを対象にします。")
    parser_db_activate.add_argument("--source", default=None, help="指定 source 名だけを対象にします。")
    parser_db_activate.add_argument("--min-quality-score", type=float, default=0.0, help="この品質スコア以上の素材だけを対象にします。")

    parser_db_deactivate = subparsers.add_parser("db-deactivate", help="条件に一致する素材を inactive に切り替えます。")
    parser_db_deactivate.add_argument("--category", default=None, help="指定カテゴリだけを対象にします。")
    parser_db_deactivate.add_argument("--source", default=None, help="指定 source 名だけを対象にします。")
    parser_db_deactivate.add_argument("--min-quality-score", type=float, default=0.0, help="この品質スコア以上の素材だけを対象にします。")

    parser_db_reset = subparsers.add_parser("db-reset", help="コーパスDBを完全に初期化(空に)します。")

    # --- 2. 学習の実行 (Training) ---
    parser_train_self = subparsers.add_parser("train-self-org", help="【推奨】SNN固有の自己組織化学習(誤差逆伝播なし)を実行します。")
    parser_train_curriculum = subparsers.add_parser(
        "train-curriculum",
        help="実データ学習カリキュラム（small/medium/large）を実行します。",
    )
    parser_train_curriculum.add_argument("--stage", choices=["small", "medium", "large"], default="small")
    parser_train_curriculum.add_argument("--dry-run", action="store_true")
    parser_train_curriculum.add_argument("--skip-gates", action="store_true")
    parser_train_curriculum.add_argument("--preflight-only", action="store_true")
    parser_train_curriculum.add_argument("--report-path", default=None, help="Optional managed report path under workspace/.")
    
    parser_train_distill = subparsers.add_parser("train-distill", help="従来の蒸留(BPベース)による学習を実行します。")
    parser_train_distill.add_argument("--model", default=model_path("sara_agent"))

    # --- 3. 推論・対話 (Inference/Chat) ---
    parser_chat_self = subparsers.add_parser("chat-self-org", help="自己組織化学習したSNNモデルと対話します。")
    
    parser_chat_distill = subparsers.add_parser("chat-distill", help="蒸留学習したモデルと対話します。")
    parser_chat_distill.add_argument("--model", default=model_path("sara_agent"))

    # --- 4. ユーティリティ ---
    parser_prune = subparsers.add_parser("prune", help="重みの低い不要な記憶を削除し、モデルを軽量化します。")
    parser_prune.add_argument("--model", default=model_path("distilled_sara_llm.msgpack"))
    parser_prune.add_argument("--threshold", type=float, default=50.0)

    parser_inspect_memory = subparsers.add_parser("inspect-memory", help="保存済みSNNメモリの健全性と retrieval diagnostics を点検します。")
    parser_inspect_memory.add_argument("--model", default=model_path("distilled_sara_llm.msgpack"))
    parser_inspect_memory.add_argument("--report", default=default_memory_health_report_path())

    parser_upgrade_memory = subparsers.add_parser("upgrade-memory", help="旧形式の SNN メモリアーティファクトを現行 managed format に再保存します。")
    parser_upgrade_memory.add_argument("--model", default=model_path("distilled_sara_llm.msgpack"))
    parser_upgrade_memory.add_argument("--output", default=default_upgraded_model_path())
    parser_upgrade_memory.add_argument("--report", default=default_upgrade_report_path())
    parser_upgrade_memory.add_argument("--replay-data", default=None, help="Optional JSONL with token sequences for rebuilding context_index.")
    parser_upgrade_memory.add_argument("--turboquant", action="store_true", help="Enable TurboQuant when saving the upgraded artifact.")

    parser_fix_memory = subparsers.add_parser("fix-memory", help="特定の direct-memory association を削除または減衰します。")
    parser_fix_memory.add_argument("--model", default=model_path("distilled_sara_llm.msgpack"))
    parser_fix_memory.add_argument("--output", default=default_fixed_model_path())
    parser_fix_memory.add_argument("--report", default=default_fix_report_path())
    parser_fix_memory.add_argument("--context-tokens", default=None, help="Context token ids separated by spaces or commas.")
    parser_fix_memory.add_argument("--context-text", default=None, help="Context text encoded by SaraTokenizer.")
    parser_fix_memory.add_argument("--wrong-token-id", type=int, default=None)
    parser_fix_memory.add_argument("--wrong-text", default=None, help="Wrong text; the last encoded token is repaired.")
    parser_fix_memory.add_argument("--tokenizer-path", default=None, help="Managed SaraTokenizer JSON path for text inputs.")
    parser_fix_memory.add_argument("--decay", type=float, default=None, help="Multiply the association weight instead of deleting it.")
    parser_fix_memory.add_argument("--dry-run", action="store_true")

    parser_build_replay = subparsers.add_parser("build-replay-data", help="既存の chat JSONL から upgrade 用 replay token JSONL を生成します。")
    parser_build_replay.add_argument("--data", default="data/raw/chat_data.jsonl")
    parser_build_replay.add_argument("--output", default=default_replay_output_path())
    parser_build_replay.add_argument("--tokenizer", default="google/gemma-2-2b")

    parser_autobot_dataset = subparsers.add_parser(
        "build-autobot-dataset",
        help="Build source-aware autobot learning materials and curriculum manifests.",
    )
    parser_autobot_dataset.add_argument(
        "--records-path",
        default=os.path.join("data", "processed", "autobot", "multimodal_records.jsonl"),
    )
    parser_autobot_dataset.add_argument(
        "--candidate-path",
        default=os.path.join("data", "interim", "autobot", "candidate_learning_materials.jsonl"),
    )
    parser_autobot_dataset.add_argument(
        "--rejected-path",
        default=os.path.join("data", "interim", "autobot", "rejected_learning_materials.jsonl"),
    )
    parser_autobot_dataset.add_argument(
        "--accepted-path",
        default=os.path.join("data", "processed", "autobot", "learning_materials.jsonl"),
    )
    parser_autobot_dataset.add_argument(
        "--curriculum-path",
        default=os.path.join("data", "processed", "autobot", "curriculum_manifest.jsonl"),
    )
    parser_autobot_dataset.add_argument(
        "--report-path",
        default=workspace_path("autobot", "dataset_builder_report.json"),
    )
    parser_autobot_dataset.add_argument(
        "--summary-path",
        default=workspace_path("autobot", "dataset_builder_summary.txt"),
    )
    parser_autobot_dataset.add_argument(
        "--fixture-request-plan-path",
        default=workspace_path("autobot", "fixture_material_request_plan.json"),
    )
    parser_autobot_dataset.add_argument(
        "--collection-targets-path",
        default=workspace_path("autobot", "dataset_builder_collection_targets.json"),
    )
    parser_autobot_dataset.add_argument("--evaluation-gap", action="append", default=None)

    parser_autobot_gap_materials = subparsers.add_parser(
        "build-autobot-gap-materials",
        help="Build source-backed gap materials from autobot collection targets.",
    )
    parser_autobot_gap_materials.add_argument(
        "--accepted-path",
        default=os.path.join("data", "processed", "autobot", "learning_materials.jsonl"),
    )
    parser_autobot_gap_materials.add_argument(
        "--targets-path",
        default=workspace_path("autobot", "dataset_builder_collection_targets.json"),
    )
    parser_autobot_gap_materials.add_argument(
        "--output-path",
        default=os.path.join("data", "processed", "autobot", "gap_materials.jsonl"),
    )
    parser_autobot_gap_materials.add_argument(
        "--report-path",
        default=workspace_path("autobot", "gap_materials_builder_report.json"),
    )
    parser_autobot_gap_materials.add_argument(
        "--summary-path",
        default=workspace_path("autobot", "gap_materials_builder_summary.txt"),
    )
    parser_autobot_gap_materials.add_argument("--blocked-request-id", action="append", default=None)
    parser_autobot_gap_materials.add_argument("--clear-blocked-request-id", action="append", default=None)
    parser_autobot_gap_enqueue = subparsers.add_parser(
        "enqueue-autobot-gap-curriculum",
        help="Enqueue gap curriculum materials into the autobot training queue.",
    )
    parser_autobot_gap_enqueue.add_argument(
        "--curriculum-path",
        default=os.path.join("data", "processed", "autobot", "gap_curriculum_manifest.jsonl"),
    )
    parser_autobot_gap_enqueue.add_argument(
        "--queue-path",
        default=workspace_path("autobot", "train_queue.json"),
    )
    parser_autobot_gap_enqueue.add_argument(
        "--report-path",
        default=workspace_path("autobot", "gap_curriculum_enqueue_report.json"),
    )
    parser_autobot_gap_enqueue.add_argument(
        "--summary-path",
        default=workspace_path("autobot", "gap_curriculum_enqueue_summary.txt"),
    )
    parser_autobot_gap_loop = subparsers.add_parser(
        "run-autobot-gap-loop",
        help="Run dataset build, gap-material generation, and gap enqueue as one managed loop.",
    )
    parser_autobot_gap_loop.add_argument(
        "--records-path",
        default=os.path.join("data", "processed", "autobot", "multimodal_records.jsonl"),
    )
    parser_autobot_gap_loop.add_argument(
        "--candidate-path",
        default=os.path.join("data", "interim", "autobot", "candidate_learning_materials.jsonl"),
    )
    parser_autobot_gap_loop.add_argument(
        "--rejected-path",
        default=os.path.join("data", "interim", "autobot", "rejected_learning_materials.jsonl"),
    )
    parser_autobot_gap_loop.add_argument(
        "--accepted-path",
        default=os.path.join("data", "processed", "autobot", "learning_materials.jsonl"),
    )
    parser_autobot_gap_loop.add_argument(
        "--curriculum-path",
        default=os.path.join("data", "processed", "autobot", "curriculum_manifest.jsonl"),
    )
    parser_autobot_gap_loop.add_argument(
        "--fixture-request-plan-path",
        default=workspace_path("autobot", "fixture_material_request_plan.json"),
    )
    parser_autobot_gap_loop.add_argument(
        "--collection-targets-path",
        default=workspace_path("autobot", "dataset_builder_collection_targets.json"),
    )
    parser_autobot_gap_loop.add_argument(
        "--gap-output-path",
        default=os.path.join("data", "processed", "autobot", "gap_materials.jsonl"),
    )
    parser_autobot_gap_loop.add_argument(
        "--gap-curriculum-path",
        default=os.path.join("data", "processed", "autobot", "gap_curriculum_manifest.jsonl"),
    )
    parser_autobot_gap_loop.add_argument(
        "--queue-path",
        default=workspace_path("autobot", "train_queue.json"),
    )
    parser_autobot_gap_loop.add_argument(
        "--report-path",
        default=workspace_path("autobot", "gap_loop_report.json"),
    )
    parser_autobot_gap_loop.add_argument(
        "--summary-path",
        default=workspace_path("autobot", "gap_loop_summary.txt"),
    )
    parser_autobot_gap_loop.add_argument("--evaluation-gap", action="append", default=None)
    parser_autobot_gap_loop.add_argument("--blocked-request-id", action="append", default=None)
    parser_autobot_gap_loop.add_argument("--clear-blocked-request-id", action="append", default=None)

    parser_autobot_gap_loop_readiness = subparsers.add_parser(
        "eval-autobot-gap-loop-readiness",
        help="Evaluate whether the managed autobot gap loop is producing usable repair curriculum.",
    )
    parser_autobot_gap_loop_readiness.add_argument(
        "--loop-report-path",
        default=workspace_path("autobot", "gap_loop_report.json"),
    )
    parser_autobot_gap_loop_readiness.add_argument(
        "--collection-targets-path",
        default=workspace_path("autobot", "dataset_builder_collection_targets.json"),
    )
    parser_autobot_gap_loop_readiness.add_argument("--dataset-report-path", default="")
    parser_autobot_gap_loop_readiness.add_argument("--gap-report-path", default="")
    parser_autobot_gap_loop_readiness.add_argument("--enqueue-report-path", default="")
    parser_autobot_gap_loop_readiness.add_argument(
        "--isolation-audit-path",
        default=workspace_path("evaluation", "phase7_isolation_audit.json"),
    )
    parser_autobot_gap_loop_readiness.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "autobot_gap_loop_readiness.json"),
    )
    parser_autobot_gap_loop_readiness.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "autobot_gap_loop_readiness_summary.txt"),
    )
    parser_autobot_gap_loop_readiness.add_argument("--min-accepted-count", type=int, default=4)
    parser_autobot_gap_loop_readiness.add_argument("--min-gap-build-coverage", type=float, default=0.5)

    parser_phase7_isolation = subparsers.add_parser(
        "eval-phase7-isolation",
        help="Audit source-aware train/evaluation isolation for Phase 7 materials.",
    )
    parser_phase7_isolation.add_argument(
        "--train-path",
        default=processed_data_path("phase7", "train.jsonl"),
    )
    parser_phase7_isolation.add_argument(
        "--evaluation-path",
        default=processed_data_path("phase7", "evaluation.jsonl"),
    )
    parser_phase7_isolation.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase7_isolation_audit.json"),
    )
    parser_phase7_isolation.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase7_isolation_audit_summary.txt"),
    )
    parser_phase7_isolation.add_argument("--max-signature-hamming-distance", type=int, default=3)

    parser_phase7_block_policy = subparsers.add_parser(
        "apply-phase7-isolation-block-policy",
        help="Apply Phase 7 isolation-audit blocks to fixture collection requests.",
    )
    parser_phase7_block_policy.add_argument(
        "--audit-path",
        default=workspace_path("evaluation", "phase7_isolation_audit.json"),
    )
    parser_phase7_block_policy.add_argument(
        "--targets-path",
        default=workspace_path("autobot", "dataset_builder_collection_targets.json"),
    )
    parser_phase7_block_policy.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase7_isolation_block_policy.json"),
    )

    parser_phase7_completion = subparsers.add_parser(
        "eval-phase7-completion",
        help="Separate Phase 7 implementation readiness from isolated-evidence completion.",
    )
    parser_phase7_completion.add_argument(
        "--readiness-path",
        default=workspace_path("evaluation", "autobot_gap_loop_readiness.json"),
    )
    parser_phase7_completion.add_argument(
        "--isolation-path",
        default=workspace_path("evaluation", "phase7_isolation_audit.json"),
    )
    parser_phase7_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase7_completion_gate.json"),
    )
    parser_phase7_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase7_completion_gate_summary.txt"),
    )

    parser_phase8_completion = subparsers.add_parser(
        "eval-phase8-completion",
        help="Separate Phase 8 implementation readiness from stronger-baseline evidence.",
    )
    parser_phase8_completion.add_argument(
        "--comparison-path",
        default=workspace_path("evaluation", "sara_ann_comparison_report.json"),
    )
    parser_phase8_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase8_completion_gate.json"),
    )
    parser_phase8_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase8_completion_gate_summary.txt"),
    )
    parser_phase8_cycle = subparsers.add_parser(
        "eval-phase8-evidence-cycle",
        help="Run the managed Phase 8 validity, ladder, comparison, and completion cycle.",
    )
    parser_phase8_cycle.add_argument("--corpus", default=processed_data_path("corpus.txt"))
    parser_phase8_cycle.add_argument("--pretrained-embedding-model", default="")
    parser_phase8_cycle.add_argument("--cross-encoder-model", default="")
    parser_phase8_cycle.add_argument("--max-docs", type=int, default=256)
    parser_phase8_cycle.add_argument("--max-cases", type=int, default=24)
    parser_phase8_cycle.add_argument("--no-history-update", action="store_true")
    parser_phase8_cycle.add_argument("--report-path", default=workspace_path("evaluation", "phase8_evidence_cycle.json"))
    parser_phase8_request = subparsers.add_parser(
        "build-phase8-reference-request",
        help="Create a managed request for a missing local Phase 8 ANN reference.",
    )
    parser_phase8_request.add_argument("--gate-path", default=workspace_path("evaluation", "phase8_completion_gate.json"))
    parser_phase8_request.add_argument("--request-path", default=workspace_path("autobot", "phase8_reference_collection_request.json"))
    parser_phase8_request.add_argument("--report-path", default=workspace_path("evaluation", "phase8_reference_collection_request.json"))

    parser_eval_external = subparsers.add_parser(
        "eval-external-validity",
        help="実データでSARA疎イベント検索とANN風密スキャン近似を比較します。",
    )
    parser_eval_external.add_argument("--corpus", default="data/processed/corpus.txt")
    parser_eval_external.add_argument("--max-docs", type=int, default=256)
    parser_eval_external.add_argument("--max-cases", type=int, default=24)
    parser_eval_external.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "real_data_external_validity.json"),
    )
    parser_eval_external.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "real_data_external_validity_summary.txt"),
    )
    parser_eval_external.add_argument(
        "--history-path",
        default=workspace_path("evaluation", "real_data_external_validity_history.json"),
    )
    parser_eval_external.add_argument("--regression-tolerance", type=float, default=0.05)
    parser_eval_external.add_argument("--pretrained-embedding-model", default="")
    parser_eval_external.add_argument("--cross-encoder-model", default="")
    parser_eval_external.add_argument("--no-history-update", action="store_true")

    parser_eval_external_ladder = subparsers.add_parser(
        "eval-external-validity-ladder",
        help="small/medium/largeの実データ外部妥当性をまとめて評価します。",
    )
    parser_eval_external_ladder.add_argument("--corpus", default="data/processed/corpus.txt")
    parser_eval_external_ladder.add_argument(
        "--profile",
        action="append",
        default=None,
        help="Profile spec as name:max_docs:max_cases. May be passed multiple times.",
    )
    parser_eval_external_ladder.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "real_data_external_validity_ladder.json"),
    )
    parser_eval_external_ladder.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "real_data_external_validity_ladder_summary.txt"),
    )
    parser_eval_external_ladder.add_argument("--regression-tolerance", type=float, default=0.05)
    parser_eval_external_ladder.add_argument("--no-history-update", action="store_true")

    parser_ann_efficiency = subparsers.add_parser(
        "eval-ann-efficiency-roadmap",
        help="ANN比のaccuracy-per-energy研究ロードマップをゲート評価します。",
    )
    parser_ann_efficiency.add_argument(
        "--energy-report-path",
        default=workspace_path("evaluation", "energy_efficiency_benchmark.json"),
    )
    parser_ann_efficiency.add_argument(
        "--external-validity-report-path",
        default=workspace_path("evaluation", "real_data_external_validity.json"),
    )
    parser_ann_efficiency.add_argument(
        "--external-ladder-report-path",
        default=workspace_path("evaluation", "real_data_external_validity_ladder.json"),
    )
    parser_ann_efficiency.add_argument(
        "--energy-measurement-report-path",
        default=workspace_path("evaluation", "energy_measurement_readiness.json"),
    )
    parser_ann_efficiency.add_argument(
        "--operational-report-path",
        default=workspace_path("release", "operational_readiness_report.json"),
    )
    parser_ann_efficiency.add_argument(
        "--output-report-path",
        default=workspace_path("evaluation", "ann_efficiency_roadmap_gate.json"),
    )
    parser_ann_efficiency.add_argument(
        "--output-summary-path",
        default=workspace_path("evaluation", "ann_efficiency_roadmap_gate_summary.txt"),
    )
    parser_ann_efficiency.add_argument("--refresh-artifacts", action="store_true")
    parser_ann_efficiency.add_argument("--allow-missing-operational", action="store_true")

    parser_sara_ann_comparison = subparsers.add_parser(
        "eval-sara-ann-comparison",
        help="SARA対ANNの比較サーフェスを研究用レポートとして生成します。",
    )
    parser_sara_ann_comparison.add_argument(
        "--external-validity-report-path",
        default=workspace_path("evaluation", "real_data_external_validity.json"),
    )
    parser_sara_ann_comparison.add_argument(
        "--external-ladder-report-path",
        default=workspace_path("evaluation", "real_data_external_validity_ladder.json"),
    )
    parser_sara_ann_comparison.add_argument(
        "--energy-measurement-report-path",
        default=workspace_path("evaluation", "energy_measurement_readiness.json"),
    )
    parser_sara_ann_comparison.add_argument(
        "--internal-maintenance-report-path",
        default=workspace_path("evaluation", "internal_maintenance_efficiency_benchmark.json"),
    )
    parser_sara_ann_comparison.add_argument(
        "--event-memory-report-path",
        default=workspace_path("evaluation", "event_memory_ingest_pipeline.json"),
    )
    parser_sara_ann_comparison.add_argument(
        "--event-memory-maintenance-coupling-report-path",
        default=workspace_path("evaluation", "event_memory_maintenance_coupling_benchmark.json"),
    )
    parser_sara_ann_comparison.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "sara_ann_comparison_report.json"),
    )
    parser_sara_ann_comparison.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "sara_ann_comparison_report.txt"),
    )

    parser_sparse_diffusion = subparsers.add_parser(
        "eval-sparse-diffusion-block-readiness",
        help="SARA互換の疎イベントDiffusion Block readinessを評価します。",
    )
    parser_sparse_diffusion.add_argument("--block-count", type=int, default=3)
    parser_sparse_diffusion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "sparse_diffusion_block_readiness.json"),
    )
    parser_sparse_diffusion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "sparse_diffusion_block_readiness_summary.txt"),
    )

    parser_rust_readiness = subparsers.add_parser(
        "eval-rust-core-readiness",
        help="Evaluate Rust sparse-runtime source readiness and optional Cargo test evidence.",
    )
    parser_rust_readiness.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "rust_core_readiness.json"),
    )
    parser_rust_readiness.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "rust_core_readiness_summary.txt"),
    )
    parser_rust_readiness.add_argument("--run-cargo-test", action="store_true")

    parser_rust_benchmark = subparsers.add_parser(
        "eval-rust-core-benchmark",
        help="Benchmark Rust sparse-runtime exports against Python reference paths.",
    )
    parser_rust_benchmark.add_argument("--iterations", type=int, default=50)
    parser_rust_benchmark.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "rust_core_benchmark.json"),
    )
    parser_rust_benchmark.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "rust_core_benchmark_summary.txt"),
    )

    parser_phase10_completion = subparsers.add_parser(
        "eval-phase10-completion",
        help="Validate the Phase 10 Rust sparse-runtime hardening evidence.",
    )
    parser_phase10_completion.add_argument(
        "--readiness-path",
        default=workspace_path("evaluation", "rust_core_readiness.json"),
    )
    parser_phase10_completion.add_argument(
        "--benchmark-path",
        default=workspace_path("evaluation", "rust_core_benchmark.json"),
    )
    parser_phase10_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase10_completion_gate.json"),
    )
    parser_phase10_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase10_completion_gate_summary.txt"),
    )

    parser_research_benchmark = subparsers.add_parser(
        "eval-research-benchmark-suite",
        help="Run the compact reproducible research benchmark suite.",
    )
    parser_research_benchmark.add_argument("--dry-run", action="store_true")
    parser_research_benchmark.add_argument("--rust-iterations", type=int, default=50)
    parser_research_benchmark.add_argument(
        "--manifest-path",
        default=workspace_path("evaluation", "research_benchmark_manifest.json"),
    )
    parser_research_benchmark.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "research_benchmark_summary.txt"),
    )

    parser_phase9_completion = subparsers.add_parser(
        "eval-phase9-completion",
        help="Validate the executed, managed Phase 9 research benchmark package.",
    )
    parser_phase9_completion.add_argument(
        "--manifest-path",
        default=workspace_path("evaluation", "research_benchmark_manifest.json"),
    )
    parser_phase9_completion.add_argument(
        "--protocol-path",
        default=os.path.join(project_root, "doc", "BENCHMARK_PROTOCOL.md"),
    )
    parser_phase9_completion.add_argument(
        "--fixture-path",
        default=os.path.join(project_root, "data", "processed", "benchmark_fixtures", "external_validity_cases.jsonl"),
    )
    parser_phase9_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase9_completion_gate.json"),
    )
    parser_phase9_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase9_completion_gate_summary.txt"),
    )

    parser_research_fixture = subparsers.add_parser(
        "eval-research-fixture-readiness",
        help="Validate repository-safe research benchmark fixtures.",
    )
    parser_research_fixture.add_argument(
        "--fixture-path",
        default=os.path.join("data", "processed", "benchmark_fixtures", "external_validity_cases.jsonl"),
    )
    parser_research_fixture.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "research_fixture_readiness.json"),
    )
    parser_research_fixture.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "research_fixture_readiness_summary.txt"),
    )

    parser_neuromorphic_matrix = subparsers.add_parser(
        "eval-neuromorphic-capability-matrix",
        help="Generate a managed neuromorphic backend capability matrix.",
    )
    parser_neuromorphic_matrix.add_argument("--profile", action="append", default=None)
    parser_neuromorphic_matrix.add_argument("--active-row-count", type=int, default=8)
    parser_neuromorphic_matrix.add_argument("--context-length", type=int, default=16)
    parser_neuromorphic_matrix.add_argument("--total-readout-size", type=int, default=64)
    parser_neuromorphic_matrix.add_argument("--quantization-bits", type=int, default=3)
    parser_neuromorphic_matrix.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "neuromorphic_capability_matrix.json"),
    )
    parser_neuromorphic_matrix.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "neuromorphic_capability_matrix_summary.txt"),
    )

    parser_phase11_completion = subparsers.add_parser(
        "eval-phase11-completion",
        help="Validate the Phase 11 neuromorphic portability evidence.",
    )
    parser_phase11_completion.add_argument(
        "--matrix-path",
        default=workspace_path("evaluation", "neuromorphic_capability_matrix.json"),
    )
    parser_phase11_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase11_completion_gate.json"),
    )
    parser_phase11_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase11_completion_gate_summary.txt"),
    )

    parser_operator_dashboard = subparsers.add_parser(
        "eval-operator-dashboard",
        help="Build the compact managed research operator dashboard.",
    )
    parser_operator_dashboard.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "operator_dashboard.json"),
    )
    parser_operator_dashboard.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "operator_dashboard_summary.txt"),
    )

    parser_phase12_completion = subparsers.add_parser(
        "eval-phase12-completion",
        help="Validate the Phase 12 operator experience surface.",
    )
    parser_phase12_completion.add_argument(
        "--dashboard-path",
        default=workspace_path("evaluation", "operator_dashboard.json"),
    )
    parser_phase12_completion.add_argument(
        "--guide-path",
        default=os.path.join(project_root, "doc", "OPERATOR_GUIDE.md"),
    )
    parser_phase12_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase12_completion_gate.json"),
    )
    parser_phase12_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase12_completion_gate_summary.txt"),
    )

    parser_phase13_completion = subparsers.add_parser(
        "eval-phase13-completion",
        help="Aggregate and validate Phase 13 sparse capability-expansion evidence.",
    )
    parser_phase13_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase13_capability_expansion.json"),
    )
    parser_phase13_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase13_capability_expansion_summary.txt"),
    )

    parser_phase14_completion = subparsers.add_parser(
        "eval-phase14-completion",
        help="Validate the Phase 14 sparse own-latent learning evidence.",
    )
    parser_phase14_completion.add_argument(
        "--benchmark-path",
        default=workspace_path("evaluation", "own_latent_learning_benchmark.json"),
    )
    parser_phase14_completion.add_argument(
        "--manifest-path",
        default=workspace_path("evaluation", "own_latent_manifest_builder.json"),
    )
    parser_phase14_completion.add_argument(
        "--fixture-path",
        default=os.path.join("data", "processed", "benchmark_fixtures", "own_latent_rhm_cases.jsonl"),
    )
    parser_phase14_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase14_completion_gate.json"),
    )
    parser_phase14_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase14_completion_gate_summary.txt"),
    )

    parser_phase15_completion = subparsers.add_parser(
        "eval-phase15-completion",
        help="Validate the Phase 15 sparse dendritic feedback evidence.",
    )
    parser_phase15_completion.add_argument("--benchmark-path", default=workspace_path("evaluation", "dendritic_feedback_gate_benchmark.json"))
    parser_phase15_completion.add_argument("--report-path", default=workspace_path("evaluation", "phase15_completion_gate.json"))
    parser_phase15_completion.add_argument("--summary-path", default=workspace_path("evaluation", "phase15_completion_gate_summary.txt"))

    parser_phase16_completion = subparsers.add_parser(
        "eval-phase16-completion",
        help="Validate the Phase 16 sparse synesthetic multimodal binding evidence.",
    )
    parser_phase16_completion.add_argument("--benchmark-path", default=workspace_path("evaluation", "synesthetic_multimodal_binding_benchmark.json"))
    parser_phase16_completion.add_argument("--report-path", default=workspace_path("evaluation", "phase16_completion_gate.json"))
    parser_phase16_completion.add_argument("--summary-path", default=workspace_path("evaluation", "phase16_completion_gate_summary.txt"))

    parser_phase17_completion = subparsers.add_parser(
        "eval-phase17-completion",
        help="Validate the Phase 17 verified sparse resonance credit evidence.",
    )
    parser_phase17_completion.add_argument("--credit-path", default=workspace_path("evaluation", "resonance_credit_benchmark.json"))
    parser_phase17_completion.add_argument("--integration-path", default=workspace_path("evaluation", "resonance_credit_integration_benchmark.json"))
    parser_phase17_completion.add_argument("--report-path", default=workspace_path("evaluation", "phase17_completion_gate.json"))
    parser_phase17_completion.add_argument("--summary-path", default=workspace_path("evaluation", "phase17_completion_gate_summary.txt"))

    parser_phase18_completion = subparsers.add_parser(
        "eval-phase18-completion",
        help="Validate the Phase 18 verified hierarchical event-state cache evidence.",
    )
    parser_phase18_completion.add_argument("--benchmark-path", default=workspace_path("evaluation", "event_state_cache_benchmark.json"))
    parser_phase18_completion.add_argument("--integration-path", default=workspace_path("evaluation", "event_state_cache_integration_benchmark.json"))
    parser_phase18_completion.add_argument("--report-path", default=workspace_path("evaluation", "phase18_completion_gate.json"))
    parser_phase18_completion.add_argument("--summary-path", default=workspace_path("evaluation", "phase18_completion_gate_summary.txt"))

    parser_phase19_completion = subparsers.add_parser(
        "eval-phase19-completion",
        help="Validate the Phase 19 sparse liquid time-constant evidence.",
    )
    parser_phase19_completion.add_argument("--benchmark-path", default=workspace_path("evaluation", "sparse_liquid_time_constant_benchmark.json"))
    parser_phase19_completion.add_argument("--report-path", default=workspace_path("evaluation", "phase19_completion_gate.json"))
    parser_phase19_completion.add_argument("--summary-path", default=workspace_path("evaluation", "phase19_completion_gate_summary.txt"))

    parser_phase20_benchmark = subparsers.add_parser(
        "eval-semantic-echo-field",
        help="Run the observed-only Phase 20 Semantic Echo Field benchmark.",
    )
    parser_phase20_benchmark.add_argument("--fixture-path", default=processed_data_path("benchmark_fixtures", "semantic_echo_field_cases.jsonl"))
    parser_phase20_benchmark.add_argument("--report-path", default=workspace_path("evaluation", "semantic_echo_field_benchmark.json"))
    parser_phase20_benchmark.add_argument("--summary-path", default=workspace_path("evaluation", "semantic_echo_field_benchmark_summary.txt"))
    parser_phase20_benchmark.add_argument("--trace-path", default=workspace_path("evaluation", "semantic_echo_field_traces.jsonl"))

    parser_phase20_completion = subparsers.add_parser(
        "eval-phase20-completion",
        help="Validate the Phase 20 Semantic Echo Field evidence.",
    )
    parser_phase20_completion.add_argument("--benchmark-path", default=workspace_path("evaluation", "semantic_echo_field_benchmark.json"))
    parser_phase20_completion.add_argument("--report-path", default=workspace_path("evaluation", "phase20_completion_gate.json"))
    parser_phase20_completion.add_argument("--summary-path", default=workspace_path("evaluation", "phase20_completion_gate_summary.txt"))

    parser_own_latent = subparsers.add_parser(
        "eval-own-latent-learning",
        help="Run the observed-only sparse own-latent learning benchmark.",
    )
    parser_own_latent.add_argument(
        "--fixture-path",
        default=os.path.join("data", "processed", "benchmark_fixtures", "own_latent_rhm_cases.jsonl"),
    )
    parser_own_latent.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "own_latent_learning_benchmark.json"),
    )
    parser_own_latent.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "own_latent_learning_benchmark_summary.txt"),
    )
    parser_own_latent.add_argument(
        "--history-path",
        default=workspace_path("evaluation", "own_latent_learning_history.json"),
    )
    parser_own_latent.add_argument("--train-sizes", default="4,8,16,32")
    parser_own_latent.add_argument("--no-history-update", action="store_true")

    parser_own_latent_manifest = subparsers.add_parser(
        "build-own-latent-manifest",
        help="Build source-backed sparse own-latent manifests from autobot materials.",
    )
    parser_own_latent_manifest.add_argument(
        "--materials-path",
        default=os.path.join("data", "processed", "autobot", "learning_materials.jsonl"),
    )
    parser_own_latent_manifest.add_argument(
        "--manifest-path",
        default=os.path.join("data", "processed", "autobot", "latent_manifest.jsonl"),
    )
    parser_own_latent_manifest.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "own_latent_manifest_builder.json"),
    )
    parser_own_latent_manifest.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "own_latent_manifest_builder_summary.txt"),
    )
    parser_own_latent_manifest.add_argument("--width", type=int, default=4096)
    parser_own_latent_manifest.add_argument("--max-events", type=int, default=32)
    parser_own_latent_manifest.add_argument("--max-terms", type=int, default=10)

    parser_dendritic_gate = subparsers.add_parser(
        "eval-dendritic-feedback-gate",
        help="Run the observed-only sparse dendritic feedback gate benchmark.",
    )
    parser_dendritic_gate.add_argument("--event-budget", type=int, default=64)
    parser_dendritic_gate.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "dendritic_feedback_gate_benchmark.json"),
    )
    parser_dendritic_gate.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "dendritic_feedback_gate_benchmark_summary.txt"),
    )

    parser_sparse_plan_trace = subparsers.add_parser(
        "eval-sparse-plan-trace-verifier",
        help="Verify sparse STRIPS-like plan traces and emit repair materials.",
    )
    parser_sparse_plan_trace.add_argument(
        "--fixture-path",
        default=os.path.join("data", "processed", "benchmark_fixtures", "sparse_plan_trace_cases.jsonl"),
    )
    parser_sparse_plan_trace.add_argument(
        "--repair-path",
        default=os.path.join("data", "processed", "autobot", "plan_trace_repair_materials.jsonl"),
    )
    parser_sparse_plan_trace.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "sparse_plan_trace_verifier.json"),
    )
    parser_sparse_plan_trace.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "sparse_plan_trace_verifier_summary.txt"),
    )

    parser_synesthetic_binding = subparsers.add_parser(
        "eval-synesthetic-multimodal-binding",
        help="Run the observed-only sparse equal-modality binding benchmark.",
    )
    parser_synesthetic_binding.add_argument(
        "--fixture-path",
        default=os.path.join(
            "data", "processed", "benchmark_fixtures", "synesthetic_multimodal_cases.jsonl"
        ),
    )
    parser_synesthetic_binding.add_argument(
        "--cross-link-path",
        default=os.path.join("data", "interim", "autobot", "synesthetic_cross_links.jsonl"),
    )
    parser_synesthetic_binding.add_argument(
        "--binding-manifest-path",
        default=os.path.join("data", "processed", "autobot", "synesthetic_binding_manifest.jsonl"),
    )
    parser_synesthetic_binding.add_argument(
        "--latent-manifest-path",
        default=os.path.join("data", "processed", "autobot", "latent_manifest.jsonl"),
    )
    parser_synesthetic_binding.add_argument(
        "--trace-path",
        default=workspace_path("evaluation", "synesthetic_multimodal_binding_traces.jsonl"),
    )
    parser_synesthetic_binding.add_argument(
        "--plug-swap-path",
        default=workspace_path("evaluation", "sparse_cortical_column_plug_swap_report.json"),
    )
    parser_synesthetic_binding.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "synesthetic_multimodal_binding_benchmark.json"),
    )
    parser_synesthetic_binding.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation", "synesthetic_multimodal_binding_benchmark_summary.txt"
        ),
    )
    parser_synesthetic_binding.add_argument("--window-ms", type=float, default=32.0)

    parser_reasoning_prior = subparsers.add_parser(
        "eval-sparse-reasoning-prior",
        help="Run the observed-only sparse future-state reasoning-prior benchmark.",
    )
    parser_reasoning_prior.add_argument(
        "--fixture-path",
        default=os.path.join(
            "data", "processed", "benchmark_fixtures", "sparse_reasoning_prior_cases.jsonl"
        ),
    )
    parser_reasoning_prior.add_argument(
        "--trace-path",
        default=workspace_path("evaluation", "sparse_reasoning_prior_traces.jsonl"),
    )
    parser_reasoning_prior.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "sparse_reasoning_prior_benchmark.json"),
    )
    parser_reasoning_prior.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "sparse_reasoning_prior_benchmark_summary.txt"),
    )

    parser_resonance_credit = subparsers.add_parser(
        "eval-resonance-credit",
        help="Run the observed-only SARA sparse resonance-credit benchmark.",
    )
    parser_resonance_credit.add_argument(
        "--fixture-path",
        default=os.path.join(
            "data", "processed", "benchmark_fixtures", "resonance_credit_cases.jsonl"
        ),
    )
    parser_resonance_credit.add_argument(
        "--trace-path",
        default=workspace_path("evaluation", "resonance_credit_traces.jsonl"),
    )
    parser_resonance_credit.add_argument(
        "--state-path",
        default=workspace_path("evaluation", "resonance_credit_state.json"),
    )
    parser_resonance_credit.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "resonance_credit_benchmark.json"),
    )
    parser_resonance_credit.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "resonance_credit_benchmark_summary.txt"),
    )

    parser_adaptive_credit = subparsers.add_parser(
        "eval-adaptive-credit-field",
        help="Run the observed-only SARA adaptive credit field benchmark.",
    )
    parser_adaptive_credit.add_argument(
        "--fixture-path",
        default=processed_data_path(
            "benchmark_fixtures",
            "adaptive_credit_field_cases.jsonl",
        ),
    )
    parser_adaptive_credit.add_argument(
        "--trace-path",
        default=workspace_path("evaluation", "adaptive_credit_field_traces.jsonl"),
    )
    parser_adaptive_credit.add_argument(
        "--state-path",
        default=workspace_path("evaluation", "adaptive_credit_field_state.json"),
    )
    parser_adaptive_credit.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "adaptive_credit_field_benchmark.json"),
    )
    parser_adaptive_credit.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "adaptive_credit_field_benchmark_summary.txt"),
    )

    parser_risa_structural = subparsers.add_parser(
        "eval-risa-structural-plasticity",
        help="Compare generic and relation-class-aware RISA structural plasticity.",
    )
    parser_risa_structural.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "risa_structural_plasticity_benchmark.json"),
    )
    parser_risa_structural.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation", "risa_structural_plasticity_benchmark_summary.txt"
        ),
    )

    parser_structural_interpolation = subparsers.add_parser(
        "eval-structural-interpolation",
        help="Run the observed-only RISA structural interpolation benchmark.",
    )
    parser_structural_interpolation.add_argument(
        "--fixture-path",
        default=processed_data_path("benchmark_fixtures", "structural_interpolation_cases.jsonl"),
    )
    parser_structural_interpolation.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "structural_interpolation_benchmark.json"),
    )
    parser_structural_interpolation.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "structural_interpolation_benchmark_summary.txt"),
    )

    parser_structural_interpolation_external = subparsers.add_parser(
        "eval-structural-interpolation-external",
        help="Run structural interpolation against the frozen independent migration manifest.",
    )
    parser_structural_interpolation_external.add_argument(
        "--manifest-path",
        default=processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl"),
    )
    parser_structural_interpolation_external.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "structural_interpolation_external_benchmark.json"),
    )
    parser_structural_interpolation_external.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "structural_interpolation_external_benchmark_summary.txt"),
    )

    parser_structural_interpolation_memory = subparsers.add_parser(
        "eval-structural-interpolation-event-memory",
        help="Evaluate structural proposals at the verified Event Memory boundary.",
    )
    parser_structural_interpolation_memory.add_argument(
        "--manifest-path",
        default=processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl"),
    )
    parser_structural_interpolation_memory.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "structural_interpolation_event_memory_benchmark.json"),
    )
    parser_structural_interpolation_memory.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "structural_interpolation_event_memory_benchmark_summary.txt"),
    )

    parser_next_level_structural = subparsers.add_parser(
        "eval-next-level-structural",
        help="Run the observed-only Phase 21 bounded structural reasoning benchmark.",
    )
    parser_next_level_structural.add_argument(
        "--fixture-path",
        default=processed_data_path("benchmark_fixtures", "next_level_structural_cases.jsonl"),
    )
    parser_next_level_structural.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "next_level_structural_benchmark.json"),
    )
    parser_next_level_structural.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "next_level_structural_benchmark_summary.txt"),
    )

    parser_continual_horizon = subparsers.add_parser(
        "eval-continual-horizon",
        help="Run the observed-only Phase 22 continual horizon benchmark.",
    )
    parser_continual_horizon.add_argument(
        "--fixture-path",
        default=processed_data_path("benchmark_fixtures", "continual_horizon_cases.jsonl"),
    )
    parser_continual_horizon.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "continual_horizon_benchmark.json"),
    )
    parser_continual_horizon.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "continual_horizon_benchmark_summary.txt"),
    )

    parser_continual_horizon_external = subparsers.add_parser(
        "eval-continual-horizon-external",
        help="Validate independent source coverage for Phase 22 horizon promotion.",
    )
    parser_continual_horizon_external.add_argument(
        "--manifest-path",
        default=processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl"),
    )
    parser_continual_horizon_external.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "continual_horizon_external_gate.json"),
    )
    parser_continual_horizon_external.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "continual_horizon_external_gate_summary.txt"),
    )

    parser_continual_horizon_request = subparsers.add_parser(
        "build-continual-horizon-collection-request",
        help="Build managed independent-data targets from the blocked Phase 22 gate.",
    )
    parser_continual_horizon_request.add_argument(
        "--gate-path",
        default=workspace_path("evaluation", "continual_horizon_external_gate.json"),
    )
    parser_continual_horizon_request.add_argument(
        "--targets-path",
        default=workspace_path("autobot", "continual_horizon_collection_targets.json"),
    )
    parser_continual_horizon_request.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "continual_horizon_collection_request.json"),
    )

    parser_phase23_fusion = subparsers.add_parser(
        "eval-phase23-structural-fusion",
        help="Run the observed-only Phase 23 structural multimodal fusion benchmark.",
    )
    parser_phase23_fusion.add_argument(
        "--fixture-path",
        default=processed_data_path("benchmark_fixtures", "phase23_structural_fusion_cases.jsonl"),
    )
    parser_phase23_fusion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase23_structural_fusion_benchmark.json"),
    )
    parser_phase23_fusion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase23_structural_fusion_benchmark_summary.txt"),
    )

    parser_phase23_external = subparsers.add_parser(
        "eval-phase23-external-multimodal",
        help="Validate independent multimodal evidence for Phase 23 promotion.",
    )
    parser_phase23_external.add_argument(
        "--manifest-path",
        default=processed_data_path("autobot", "phase23_independent_multimodal_manifest.jsonl"),
    )
    parser_phase23_external.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase23_external_multimodal_gate.json"),
    )
    parser_phase23_external.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase23_external_multimodal_gate_summary.txt"),
    )

    parser_phase23_request = subparsers.add_parser(
        "build-phase23-multimodal-collection-request",
        help="Build managed independent multimodal collection targets from the Phase 23 gate.",
    )
    parser_phase23_request.add_argument(
        "--gate-path",
        default=workspace_path("evaluation", "phase23_external_multimodal_gate.json"),
    )
    parser_phase23_request.add_argument(
        "--targets-path",
        default=workspace_path("autobot", "phase23_multimodal_collection_targets.json"),
    )
    parser_phase23_request.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase23_multimodal_collection_request.json"),
    )

    parser_phase24_causal = subparsers.add_parser(
        "eval-phase24-causal",
        help="Run the observed-only Phase 24 causal and counterfactual benchmark.",
    )
    parser_phase24_causal.add_argument(
        "--fixture-path",
        default=processed_data_path("benchmark_fixtures", "phase24_causal_cases.jsonl"),
    )
    parser_phase24_causal.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase24_causal_benchmark.json"),
    )
    parser_phase24_causal.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase24_causal_benchmark_summary.txt"),
    )

    parser_phase25_agent = subparsers.add_parser(
        "eval-phase25-agent-loop",
        help="Run the observed-only Phase 25 bounded agent-loop benchmark.",
    )
    parser_phase25_agent.add_argument(
        "--fixture-path",
        default=processed_data_path("benchmark_fixtures", "phase25_agent_cases.jsonl"),
    )
    parser_phase25_agent.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase25_agent_loop_benchmark.json"),
    )
    parser_phase25_agent.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase25_agent_loop_benchmark_summary.txt"),
    )

    parser_next_level_review = subparsers.add_parser(
        "eval-next-level-promotion-review",
        help="Review Phase 21-25 evidence without self-promoting defaults.",
    )
    parser_next_level_review.add_argument("--evaluation-dir", default=workspace_path("evaluation"))
    parser_next_level_review.add_argument("--report-path", default=workspace_path("evaluation", "next_level_promotion_review.json"))
    parser_next_level_review.add_argument("--gate-path", default=workspace_path("evaluation", "next_level_promotion_gate.json"))
    parser_next_level_review.add_argument("--journal-path", default=workspace_path("evaluation", "next_level_research_journal.jsonl"))
    parser_next_level_review.add_argument("--approval-path", default=workspace_path("evaluation", "next_level_human_approval.json"))

    parser_next_level_approval = subparsers.add_parser(
        "record-next-level-human-approval",
        help="Record evidence-bound human approval for the next-level promotion review.",
    )
    parser_next_level_approval.add_argument("--evaluation-dir", default=workspace_path("evaluation"))
    parser_next_level_approval.add_argument("--reviewer", required=True)
    parser_next_level_approval.add_argument("--note", default="")
    parser_next_level_approval.add_argument("--output-path", default=workspace_path("evaluation", "next_level_human_approval.json"))

    parser_scale_up_readiness = subparsers.add_parser(
        "eval-scale-up-readiness",
        help="Prepare, but do not run, the larger post-roadmap experiment.",
    )
    parser_scale_up_readiness.add_argument("--promotion-gate", default=workspace_path("evaluation", "next_level_promotion_gate.json"))
    parser_scale_up_readiness.add_argument("--external-gate", default=workspace_path("evaluation", "continual_horizon_external_gate.json"))
    parser_scale_up_readiness.add_argument("--multimodal-gate", default=workspace_path("evaluation", "phase23_external_multimodal_gate.json"))
    parser_scale_up_readiness.add_argument("--preregistration-path", default=workspace_path("evaluation", "scale_up_preregistration.json"))
    parser_scale_up_readiness.add_argument("--output-path", default=workspace_path("evaluation", "scale_up_experiment_readiness.json"))

    parser_scale_up_preregistration = subparsers.add_parser(
        "register-scale-up-preregistration",
        help="Register an immutable managed Phase 29 experiment protocol.",
    )
    parser_scale_up_preregistration.add_argument("--draft-path", required=True)
    parser_scale_up_preregistration.add_argument(
        "--output-path",
        default=workspace_path(
            "evaluation", "scale_up_preregistration.json"
        ),
    )

    parser_phase27_runtime = subparsers.add_parser(
        "eval-phase27-portable-runtime",
        help="Check canonical sparse IR portability readiness without claiming Rust equivalence.",
    )
    parser_phase27_runtime.add_argument("--output-path", default=workspace_path("evaluation", "phase27_portable_runtime_readiness.json"))
    parser_phase27_runtime.add_argument("--rust-report-path", default=workspace_path("evaluation", "rust_core_benchmark.json"))
    parser_phase27_runtime.add_argument("--tokenizer-report-path", default=workspace_path("evaluation", "phase27_tokenizer_acceleration_benchmark.json"))

    parser_phase27_tokenizer = subparsers.add_parser(
        "eval-phase27-tokenizer-acceleration",
        help="Evaluate bounded exact tokenization without production promotion.",
    )
    parser_phase27_tokenizer.add_argument(
        "--fixture-path",
        default=processed_data_path(
            "benchmark_fixtures",
            "phase27_tokenizer_conformance_cases.jsonl",
        ),
    )
    parser_phase27_tokenizer.add_argument(
        "--output-path",
        default=workspace_path(
            "evaluation",
            "phase27_tokenizer_acceleration_benchmark.json",
        ),
    )

    parser_phase31_repetition = subparsers.add_parser(
        "eval-phase31-repetition-consolidation",
        help="Evaluate bounded repetition-dependent memory consolidation.",
    )
    parser_phase31_repetition.add_argument(
        "--fixture-path",
        default=processed_data_path(
            "benchmark_fixtures",
            "phase31_repetition_consolidation_cases.jsonl",
        ),
    )
    parser_phase31_repetition.add_argument(
        "--output-path",
        default=workspace_path(
            "evaluation",
            "phase31_repetition_consolidation_benchmark.json",
        ),
    )

    parser_phase31_reranking = subparsers.add_parser(
        "eval-phase31-repetition-reranking",
        help="Evaluate candidate-only repetition reranking.",
    )
    parser_phase31_reranking.add_argument(
        "--fixture-path",
        default=processed_data_path(
            "benchmark_fixtures",
            "phase31_repetition_reranking_cases.jsonl",
        ),
    )
    parser_phase31_reranking.add_argument(
        "--output-path",
        default=workspace_path(
            "evaluation",
            "phase31_repetition_reranking_benchmark.json",
        ),
    )

    parser_phase33_preregistration = subparsers.add_parser(
        "register-phase33-structured-edge-preregistration",
        help="Register an immutable managed Phase 33 experiment protocol.",
    )
    parser_phase33_preregistration.add_argument("--draft-path", required=True)
    parser_phase33_preregistration.add_argument(
        "--output-path",
        default=workspace_path(
            "evaluation",
            "phase33_structured_edge_preregistration.json",
        ),
    )
    parser_phase33_draft = subparsers.add_parser(
        "build-phase33-structured-edge-preregistration-draft",
        help="Freeze the Phase 33 fixture and CPU environment fingerprints.",
    )
    parser_phase33_draft.add_argument(
        "--fixture-path",
        default=processed_data_path(
            "benchmark_fixtures",
            "phase33_structured_edge_cases.jsonl",
        ),
    )
    parser_phase33_draft.add_argument(
        "--draft-path",
        default=workspace_path(
            "evaluation",
            "phase33_structured_edge_preregistration_draft.json",
        ),
    )
    parser_phase33_draft.add_argument(
        "--environment-path",
        default=workspace_path(
            "evaluation",
            "phase33_structured_edge_environment.json",
        ),
    )

    parser_level2_matrix = subparsers.add_parser(
        "eval-level2-capability-matrix",
        help="Build the Level-2 capability matrix without promotion.",
    )
    parser_level2_matrix.add_argument("--evaluation-dir", default=workspace_path("evaluation"))
    parser_level2_matrix.add_argument("--output-path", default=workspace_path("evaluation", "level2_capability_matrix.json"))
    parser_level2_matrix.add_argument("--summary-path", default=workspace_path("evaluation", "level2_capability_matrix_summary.txt"))

    parser_adaptive_credit_event_memory = subparsers.add_parser(
        "eval-adaptive-credit-event-memory",
        help="Run the observed-only adaptive credit/Event Memory integration benchmark.",
    )
    parser_adaptive_credit_event_memory.add_argument(
        "--fixture-path",
        default=processed_data_path(
            "benchmark_fixtures",
            "adaptive_credit_event_memory_cases.jsonl",
        ),
    )
    parser_adaptive_credit_event_memory.add_argument(
        "--trace-path",
        default=workspace_path("evaluation", "adaptive_credit_event_memory_traces.jsonl"),
    )
    parser_adaptive_credit_event_memory.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "adaptive_credit_event_memory_benchmark.json"),
    )
    parser_adaptive_credit_event_memory.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "adaptive_credit_event_memory_benchmark_summary.txt",
        ),
    )

    parser_resonance_integration = subparsers.add_parser(
        "eval-resonance-credit-integration",
        help="Bridge managed SARA evidence reports into observed-only resonance credit.",
    )
    parser_resonance_integration.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "resonance_credit_integration_benchmark.json"),
    )
    parser_resonance_integration.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation", "resonance_credit_integration_benchmark_summary.txt"
        ),
    )
    parser_resonance_integration.add_argument(
        "--trace-path",
        default=workspace_path("evaluation", "resonance_credit_integration_traces.jsonl"),
    )

    parser_event_state_cache = subparsers.add_parser(
        "eval-event-state-cache",
        help="Run the observed-only verified hierarchical event-state cache benchmark.",
    )
    parser_event_state_cache.add_argument(
        "--fixture-path",
        default=processed_data_path(
            "benchmark_fixtures",
            "event_state_cache_cases.jsonl",
        ),
    )
    parser_event_state_cache.add_argument(
        "--candidate-path",
        default=interim_data_path(
            "event_state_cache",
            "candidates.jsonl",
        ),
    )
    parser_event_state_cache.add_argument(
        "--manifest-path",
        default=processed_data_path(
            "event_state_cache",
            "manifest.jsonl",
        ),
    )
    parser_event_state_cache.add_argument(
        "--trace-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_traces.jsonl",
        ),
    )
    parser_event_state_cache.add_argument(
        "--state-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_state.json",
        ),
    )
    parser_event_state_cache.add_argument(
        "--report-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_benchmark.json",
        ),
    )
    parser_event_state_cache.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_benchmark_summary.txt",
        ),
    )

    parser_event_state_cache_integration = subparsers.add_parser(
        "eval-event-state-cache-integration",
        help="Evaluate source-aware event-state caching with managed resonance evidence.",
    )
    parser_event_state_cache_integration.add_argument(
        "--manifest-path",
        default=processed_data_path("autobot", "latent_manifest.jsonl"),
    )
    parser_event_state_cache_integration.add_argument(
        "--trace-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_integration_traces.jsonl",
        ),
    )
    parser_event_state_cache_integration.add_argument(
        "--round-trip-state-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_round_trip_state.json",
        ),
    )
    parser_event_state_cache_integration.add_argument(
        "--report-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_integration_benchmark.json",
        ),
    )
    parser_event_state_cache_integration.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "event_state_cache_integration_benchmark_summary.txt",
        ),
    )

    parser_architecture_migration = subparsers.add_parser(
        "eval-architecture-migration",
        help="Run the frozen source-isolated Event Memory architecture-migration benchmark.",
    )
    parser_architecture_migration.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "architecture_migration_benchmark.json"),
    )
    parser_architecture_migration.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation", "architecture_migration_benchmark_summary.txt"
        ),
    )

    parser_architecture_migration_external = subparsers.add_parser(
        "eval-architecture-migration-external",
        help="Gate architecture migration on provenance-qualified independent external sources.",
    )
    parser_architecture_migration_external.add_argument(
        "--manifest-path",
        default=processed_data_path("autobot", "latent_manifest.jsonl"),
    )
    parser_architecture_migration_external.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "architecture_migration_external_gate.json"),
    )
    parser_architecture_migration_external.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation", "architecture_migration_external_gate_summary.txt"
        ),
    )
    parser_architecture_migration_request = subparsers.add_parser(
        "build-architecture-migration-collection-request",
        help="Convert blocked architecture-migration evidence into collection targets.",
    )
    parser_architecture_migration_request.add_argument("--gate-path", default=workspace_path("evaluation", "architecture_migration_external_gate.json"))
    parser_architecture_migration_request.add_argument("--targets-path", default=workspace_path("autobot", "architecture_migration_collection_targets.json"))
    parser_architecture_migration_request.add_argument("--report-path", default=workspace_path("evaluation", "architecture_migration_collection_request.json"))
    parser_architecture_migration_manifest = subparsers.add_parser("build-architecture-migration-manifest", help="Qualify external latent records for architecture-migration evaluation.")
    parser_architecture_migration_manifest.add_argument("--input-path", default=processed_data_path("autobot", "latent_manifest.jsonl"))
    parser_architecture_migration_manifest.add_argument("--output-path", default=processed_data_path("autobot", "architecture_migration_external_manifest.jsonl"))
    parser_architecture_migration_manifest.add_argument("--report-path", default=workspace_path("evaluation", "architecture_migration_manifest_builder.json"))
    parser_architecture_migration_cycle = subparsers.add_parser("eval-architecture-migration-evidence-cycle", help="Run architecture-migration qualification, gate, and collection handoff.")
    parser_architecture_migration_cycle.add_argument("--input-path", default=processed_data_path("autobot", "latent_manifest.jsonl"))
    parser_architecture_migration_cycle.add_argument("--report-path", default=workspace_path("evaluation", "architecture_migration_evidence_cycle.json"))

    parser_event_memory_ingest = subparsers.add_parser(
        "eval-event-memory-ingest-pipeline",
        help="Run the bounded Event Memory ingest loop smoke benchmark.",
    )
    parser_event_memory_ingest.add_argument(
        "--report-path",
        default=workspace_path(
            "evaluation",
            "event_memory_ingest_pipeline.json",
        ),
    )
    parser_event_memory_ingest.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "event_memory_ingest_pipeline_summary.txt",
        ),
    )
    parser_event_memory_maintenance_coupling = subparsers.add_parser(
        "eval-event-memory-maintenance-coupling",
        help="Run the Event Memory compression versus self-state maintenance coupling benchmark.",
    )
    parser_event_memory_maintenance_coupling.add_argument(
        "--report-path",
        default=workspace_path(
            "evaluation",
            "event_memory_maintenance_coupling_benchmark.json",
        ),
    )
    parser_event_memory_maintenance_coupling.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "event_memory_maintenance_coupling_benchmark_summary.txt",
        ),
    )

    parser_persistent_self_state = subparsers.add_parser(
        "eval-persistent-self-state",
        help="Run the bounded persistent self-state benchmark.",
    )
    parser_persistent_self_state.add_argument(
        "--report-path",
        default=workspace_path(
            "evaluation",
            "persistent_self_state_benchmark.json",
        ),
    )
    parser_persistent_self_state.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "persistent_self_state_benchmark_summary.txt",
        ),
    )
    parser_idle_replay = subparsers.add_parser(
        "eval-idle-replay",
        help="Run the bounded idle replay benchmark.",
    )
    parser_idle_replay.add_argument(
        "--report-path",
        default=workspace_path(
            "evaluation",
            "idle_replay_benchmark.json",
        ),
    )
    parser_idle_replay.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "idle_replay_benchmark_summary.txt",
        ),
    )
    parser_internal_maintenance = subparsers.add_parser(
        "eval-internal-maintenance-efficiency",
        help="Run the bounded internal maintenance efficiency benchmark.",
    )
    parser_internal_maintenance.add_argument(
        "--report-path",
        default=workspace_path(
            "evaluation",
            "internal_maintenance_efficiency_benchmark.json",
        ),
    )
    parser_internal_maintenance.add_argument(
        "--summary-path",
        default=workspace_path(
            "evaluation",
            "internal_maintenance_efficiency_benchmark_summary.txt",
        ),
    )
    parser_internal_integration = subparsers.add_parser(
        "eval-internal-practical-integration",
        help="Run the internal-only practical integration benchmark.",
    )
    parser_internal_integration.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "internal_practical_integration_benchmark.json"),
    )
    parser_internal_integration.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "internal_practical_integration_benchmark_summary.txt"),
    )

    parser_operator_llm = subparsers.add_parser(
        "eval-operator-llm-assistant-readiness",
        help="Evaluate the optional local LLM operator-assistant proposal gate.",
    )
    parser_operator_llm.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "operator_llm_assistant_readiness.json"),
    )
    parser_operator_llm.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "operator_llm_assistant_readiness_summary.txt"),
    )
    parser_operator_llm.add_argument(
        "--enabled",
        action="store_true",
        help="Model the assistant as enabled. Readiness should fail because default must stay disabled.",
    )

    parser_record_energy = subparsers.add_parser(
        "record-energy-measurement",
        help="SARA/ANNの実測joule-per-success用JSONLをdata/rawへ追記します。",
    )
    parser_record_energy.add_argument("--measurement-path", default="data/raw/energy_measurements.jsonl")
    parser_record_energy.add_argument("--run-id", required=True)
    parser_record_energy.add_argument("--system", choices=["sara", "ann"], required=True)
    parser_record_energy.add_argument("--task", required=True)
    parser_record_energy.add_argument("--success-count", type=int, required=True)
    parser_record_energy.add_argument("--joules", type=float, default=0.0)
    parser_record_energy.add_argument("--source", default="manual")
    parser_record_energy.add_argument("--duration-seconds", type=float, default=None)
    parser_record_energy.add_argument("--average-watts", type=float, default=None)
    parser_record_energy.add_argument("--session-id", default="ann-efficiency-real-joule")
    parser_record_energy.add_argument("--notes", default="")
    parser_record_energy.add_argument(
        "--protocol-version",
        default="sara-energy-fair-comparison-v2",
    )
    parser_record_energy.add_argument("--pair-id", required=True)
    parser_record_energy.add_argument("--replicate-index", type=int, required=True)
    parser_record_energy.add_argument("--environment-fingerprint", default="")
    parser_record_energy.add_argument("--task-fixture-hash", required=True)
    parser_record_energy.add_argument("--success-criterion-id", required=True)
    parser_record_energy.add_argument("--measurement-boundary", required=True)
    parser_record_energy.add_argument("--measurement-tool", required=True)
    parser_record_energy.add_argument("--cpu-model", required=True)
    parser_record_energy.add_argument("--thread-count", type=int, required=True)
    parser_record_energy.add_argument("--process-affinity", required=True)
    parser_record_energy.add_argument("--power-mode", required=True)
    parser_record_energy.add_argument("--warmup-count", type=int, required=True)
    parser_record_energy.add_argument("--measured-repetitions", type=int, required=True)
    parser_record_energy.add_argument("--trial-count", type=int, required=True)
    parser_record_energy.add_argument("--run-order", type=int, choices=[1, 2], required=True)
    parser_record_energy.add_argument("--maintenance-selected-count", type=int, default=None)
    parser_record_energy.add_argument("--maintenance-phase-count", type=int, default=None)
    parser_record_energy.add_argument("--maintenance-refresh-count", type=int, default=None)
    parser_record_energy.add_argument("--maintenance-event-cost", type=float, default=None)
    parser_record_energy.add_argument("--max-success-rate-delta", type=float, default=0.0)
    parser_record_energy.add_argument(
        "--min-paired-replicates-per-task",
        type=int,
        default=3,
    )
    parser_record_energy.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "energy_measurement_readiness.json"),
    )
    parser_record_energy.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "energy_measurement_readiness_summary.txt"),
    )
    parser_record_energy.add_argument(
        "--session-plan-path",
        default=workspace_path("evaluation", "energy_measurement_session_plan.json"),
    )
    parser_record_energy.add_argument(
        "--session-plan-summary-path",
        default=workspace_path("evaluation", "energy_measurement_session_plan.txt"),
    )

    parser_phase6_completion = subparsers.add_parser(
        "eval-phase6-completion",
        help="Classify Phase 6 implementation readiness and physical-evidence completion.",
    )
    parser_phase6_completion.add_argument(
        "--readiness-path",
        default=workspace_path("evaluation", "energy_measurement_readiness.json"),
    )
    parser_phase6_completion.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase6_completion_gate.json"),
    )
    parser_phase6_completion.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase6_completion_gate_summary.txt"),
    )

    parser_physical_pair = subparsers.add_parser(
        "run-physical-energy-pair",
        help="Run or plan one fair paired SARA/ANN physical-energy workload.",
    )
    parser_physical_pair.add_argument("--pair-id", required=True)
    parser_physical_pair.add_argument("--replicate-index", type=int, required=True)
    parser_physical_pair.add_argument("--corpus-path", default="data/processed/corpus.txt")
    parser_physical_pair.add_argument("--max-docs", type=int, default=256)
    parser_physical_pair.add_argument("--max-cases", type=int, default=24)
    parser_physical_pair.add_argument("--repetitions", type=int, default=10000)
    parser_physical_pair.add_argument("--warmup-count", type=int, default=2)
    parser_physical_pair.add_argument("--thread-count", type=int, default=1)
    parser_physical_pair.add_argument("--process-affinity", default="unbound-single-process")
    parser_physical_pair.add_argument("--power-mode", default="ac-power-default")
    parser_physical_pair.add_argument(
        "--measurement-tool",
        default="external-meter-manual-v1",
    )
    parser_physical_pair.add_argument("--sara-joules", type=float, default=0.0)
    parser_physical_pair.add_argument("--ann-joules", type=float, default=0.0)
    parser_physical_pair.add_argument(
        "--auto-system-energy-estimate",
        action="store_true",
        help="Estimate energy from macOS ioreg telemetry; this is not physical-meter evidence.",
    )
    parser_physical_pair.add_argument(
        "--measurement-path",
        default="data/raw/energy_measurements.jsonl",
    )
    parser_physical_pair.add_argument(
        "--meter-reading-path",
        default="",
        help="JSON file with measured SARA/ANN joules or average watts and duration.",
    )
    parser_physical_pair.add_argument(
        "--manifest-path",
        default=workspace_path("evaluation", "physical_energy_pair_manifest.json"),
    )
    parser_physical_pair.add_argument(
        "--trace-path",
        default=workspace_path("evaluation", "physical_energy_pair_trace.jsonl"),
    )
    parser_physical_pair.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "physical_energy_pair_report.json"),
    )
    parser_physical_pair.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "physical_energy_pair_summary.txt"),
    )
    parser_physical_pair.add_argument(
        "--meter-template-path",
        default=workspace_path("evaluation", "physical_energy_pair_meter_template.json"),
    )
    parser_physical_pair.add_argument("--dry-run", action="store_true")

    parser_physical_session_batch = subparsers.add_parser(
        "run-physical-energy-session-batch",
        help="Build or execute a thin batch plan for physical-energy pair sessions.",
    )
    parser_physical_session_batch.add_argument(
        "--session-plan-path",
        default=workspace_path("evaluation", "energy_measurement_session_plan.json"),
    )
    parser_physical_session_batch.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "physical_energy_session_batch.json"),
    )
    parser_physical_session_batch.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "physical_energy_session_batch.txt"),
    )
    parser_physical_session_batch.add_argument("--execute-dry-run-pairs", action="store_true")

    parser_physical_session_progress = subparsers.add_parser(
        "eval-physical-energy-session-progress",
        help="Summarize progress for a physical-energy measurement session.",
    )
    parser_physical_session_progress.add_argument(
        "--batch-report-path",
        default=workspace_path("evaluation", "physical_energy_session_batch.json"),
    )
    parser_physical_session_progress.add_argument(
        "--measurement-path",
        default="data/raw/energy_measurements.jsonl",
    )
    parser_physical_session_progress.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "physical_energy_session_progress.json"),
    )
    parser_physical_session_progress.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "physical_energy_session_progress.txt"),
    )
    parser_physical_session_progress.add_argument(
        "--internal-maintenance-report-path",
        default=workspace_path("evaluation", "internal_maintenance_efficiency_benchmark.json"),
    )
    parser_physical_session_progress.add_argument(
        "--event-memory-maintenance-coupling-report-path",
        default=workspace_path("evaluation", "event_memory_maintenance_coupling_benchmark.json"),
    )

    parser_clean = subparsers.add_parser("clean", help="中間データを削除して環境をリセットします。")

    args = parser.parse_args()
    db_path = "data/sara_corpus.db"

    if args.command == "db-import":
        db = SaraCorpusDB(db_path)
        print(f"[DB] {args.file} をインポートしています...")
        added = db.import_file(
            args.file,
            category=args.category,
            lang=args.lang,
            source_version=args.source_version,
            quality_score=args.quality_score,
            is_active=not args.inactive,
        )
        summary = db.get_material_summary()
        review_summary = db.get_review_summary()
        import_report = {
            "file": args.file,
            "added_count": int(added),
            "metadata": {
                "category": args.category,
                "lang": args.lang,
                "source_version": args.source_version,
                "quality_score": float(args.quality_score),
                "is_active": not args.inactive,
            },
            "summary": summary,
            "review_summary": review_summary,
        }
        if args.report:
            report_path = ensure_parent_directory(args.report)
        else:
            report_path = ensure_parent_directory(workspace_path("reports", "db_import_report.json"))
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(import_report, handle, indent=2, ensure_ascii=False)
        print(f"Saved import report: {report_path}")
        print(f"✅ {added} 件のデータを新しくDBに登録しました。")

    elif args.command == "db-status":
        if not os.path.exists(db_path):
            print("DBが存在しません。まだデータが登録されていません。")
        else:
            db = SaraCorpusDB(db_path)
            stats = db.get_stats()
            summary = db.get_material_summary()
            review_summary = db.get_review_summary()
            if args.format == "json":
                print(
                    json.dumps(
                        {
                            "stats": [{"text_type": t_type, "count": count} for t_type, count in stats],
                            "summary": summary,
                            "review_summary": review_summary,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return
            print("=== SARA Corpus Database Status ===")
            total = 0
            for t_type, count in stats:
                print(f"- {t_type.capitalize()} データ: {count} 件")
                total += count
            print(f"合計: {total} 件")
            print(f"有効素材: {summary['active_count']} 件")
            print(f"無効素材: {summary['inactive_count']} 件")
            print(f"平均品質スコア: {summary['avg_quality_score']:.2f}")
            if summary["categories"]:
                print("カテゴリ内訳:")
                for category, count in summary["categories"]:
                    print(f"- {category}: {count} 件")
            if review_summary["by_source"]:
                print("source内訳:")
                for item in review_summary["by_source"][:5]:
                    print(f"- {item['key']}: {item['count']} 件 (avg_quality={item['avg_quality_score']:.2f})")
            if review_summary["by_lang"]:
                print("lang内訳:")
                for item in review_summary["by_lang"]:
                    print(f"- {item['key']}: {item['count']} 件")

    elif args.command == "db-list":
        if not os.path.exists(db_path):
            print("DBが存在しません。まだデータが登録されていません。")
        else:
            db = SaraCorpusDB(db_path)
            materials = db.list_materials(
                category=args.category,
                source=args.source,
                min_quality_score=args.min_quality_score,
                show_inactive=args.show_inactive,
                limit=args.limit,
            )
            if args.format == "json":
                print(json.dumps({"items": materials}, indent=2, ensure_ascii=False))
                return
            print("=== SARA Corpus Material Preview ===")
            if not materials:
                print("該当する素材はありません。")
            for idx, item in enumerate(materials, start=1):
                status_label = "active" if item["is_active"] else "inactive"
                print(
                    f"{idx}. [{item['text_type']}/{item['category']}] "
                    f"q={item['quality_score']:.2f} lang={item['lang']} "
                    f"source={item['source']} version={item['source_version'] or '-'} "
                    f"status={status_label}"
                )
                print(f"   {item['preview']}")

    elif args.command == "db-export":
        db = SaraCorpusDB(db_path)
        material_summary = db.get_material_summary()
        review_summary = db.get_review_summary()
        plan = db.summarize_export_plan(
            category=args.category,
            source=args.source,
            min_quality_score=args.min_quality_score,
            show_inactive=args.show_inactive,
        )
        total_material_count = int(material_summary.get("total_count", 0) or 0)
        selected_count = int(plan.get("total_count", 0) or 0)
        export_report = {
            "filters": {
                "category": args.category,
                "source": args.source,
                "min_quality_score": float(args.min_quality_score),
                "show_inactive": bool(args.show_inactive),
            },
            "dry_run": bool(args.dry_run),
            "material_summary": material_summary,
            "review_summary": review_summary,
            "plan": plan,
            "delta": {
                "selected_count": selected_count,
                "total_material_count": total_material_count,
                "selected_ratio": (
                    float(selected_count) / float(total_material_count)
                    if total_material_count > 0
                    else 0.0
                ),
            },
        }
        if args.dry_run:
            print("=== SARA Export Dry Run ===")
            print(f"total_count: {plan['total_count']}")
            for item in plan["items"]:
                print(
                    f"- {item['text_type']}/{item['category']}: "
                    f"{item['count']} 件 (avg_quality={item['avg_quality_score']:.2f})"
                )
            if args.report:
                report_path = ensure_parent_directory(args.report)
                with open(report_path, "w", encoding="utf-8") as handle:
                    json.dump(export_report, handle, indent=2, ensure_ascii=False)
                print(f"Saved export report: {report_path}")
            return
        print("[DB] 自己組織化学習用コーパス(corpus.txt)を出力しています...")
        c_count = db.export_for_self_organized(
            "data/processed/corpus.txt",
            category=args.category,
            source=args.source,
            min_quality_score=args.min_quality_score,
            show_inactive=args.show_inactive,
        )
        print(f"  -> {c_count} 件エクスポート完了")
        
        print("[DB] 蒸留学習用データ(chat_data.jsonl)を出力しています...")
        d_count = db.export_for_distillation(
            "data/raw/chat_data.jsonl",
            category=args.category,
            source=args.source,
            min_quality_score=args.min_quality_score,
            show_inactive=args.show_inactive,
        )
        print(f"  -> {d_count} 件エクスポート完了")
        export_report["outputs"] = {
            "corpus_path": "data/processed/corpus.txt",
            "chat_data_path": "data/raw/chat_data.jsonl",
            "corpus_count": int(c_count),
            "chat_count": int(d_count),
        }
        if args.report:
            report_path = ensure_parent_directory(args.report)
        else:
            report_path = ensure_parent_directory(workspace_path("reports", "db_export_report.json"))
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(export_report, handle, indent=2, ensure_ascii=False)
        print(f"Saved export report: {report_path}")
        print("✅ エクスポートが完了しました。学習を開始できます。")

    elif args.command == "db-activate":
        db = SaraCorpusDB(db_path)
        updated = db.set_material_active_state(
            True,
            category=args.category,
            source=args.source,
            min_quality_score=args.min_quality_score,
            include_inactive=True,
        )
        print(f"✅ {updated} 件の素材を active に切り替えました。")

    elif args.command == "db-deactivate":
        db = SaraCorpusDB(db_path)
        updated = db.set_material_active_state(
            False,
            category=args.category,
            source=args.source,
            min_quality_score=args.min_quality_score,
            include_inactive=True,
        )
        print(f"✅ {updated} 件の素材を inactive に切り替えました。")

    elif args.command == "db-reset":
        if os.path.exists(db_path):
            os.remove(db_path)
            print("🗑️ データベースを初期化しました。")
        else:
            print("データベースは既に空です。")

    elif args.command == "train-self-org":
        print("🧠 自己組織化学習(Self-Organized SNN)を開始します...")
        result = subprocess.run([sys.executable, "scripts/train/train_self_organized.py"])
        sys.exit(result.returncode)

    elif args.command == "train-curriculum":
        print(f"🧪 実データ学習カリキュラムを開始します... stage={args.stage}")
        command = [
            sys.executable,
            "scripts/train/run_real_data_curriculum.py",
            "--stage",
            str(args.stage),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.skip_gates:
            command.append("--skip-gates")
        if args.preflight_only:
            command.append("--preflight-only")
        if args.report_path:
            command.extend(["--report-path", str(args.report_path)])
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "train-distill":
        print("🔥 蒸留学習(Distillation)を開始します...")
        from train.train_chat import train_chat_model
        train_chat_model("data/raw/chat_data.jsonl", save_dir=args.model)

    elif args.command == "chat-self-org":
        subprocess.run([sys.executable, "scripts/eval/chat_self_organized.py"])

    elif args.command == "chat-distill":
        subprocess.run([sys.executable, "scripts/eval/chat_agent.py", "--model-dir", args.model])

    elif args.command == "prune":
        prune_model_memory(args.model, args.threshold)

    elif args.command == "inspect-memory":
        report = inspect_inference_memory(args.model, args.report)
        print("=== SARA Memory Health Report ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.command == "upgrade-memory":
        report = upgrade_inference_memory(
            args.model,
            args.output,
            replay_data_path=args.replay_data,
            enable_turboquant=args.turboquant,
        )
        if args.report:
            report_path = ensure_parent_directory(args.report)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
            report["report_path"] = report_path
        print("=== SARA Memory Upgrade Report ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.command == "fix-memory":
        context_tokens = None
        if args.context_tokens:
            context_tokens = [
                int(item.strip())
                for item in str(args.context_tokens).replace(",", " ").split()
                if item.strip()
            ]
        report = fix_inference_memory(
            args.model,
            args.output,
            context_tokens=context_tokens,
            context_text=args.context_text,
            wrong_token_id=args.wrong_token_id,
            wrong_text=args.wrong_text,
            tokenizer_path=args.tokenizer_path,
            decay=args.decay,
            dry_run=args.dry_run,
            report_path=args.report,
        )
        print("=== SARA Memory Fix Report ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.command == "build-replay-data":
        report = build_replay_data(args.data, args.output, tokenizer_name=args.tokenizer)
        print("=== SARA Replay Data Report ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.command == "build-autobot-dataset":
        command = [
            sys.executable,
            "bot/dataset_builder.py",
            "--records-path",
            str(args.records_path),
            "--candidate-path",
            str(args.candidate_path),
            "--rejected-path",
            str(args.rejected_path),
            "--accepted-path",
            str(args.accepted_path),
            "--curriculum-path",
            str(args.curriculum_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--fixture-request-plan-path",
            str(args.fixture_request_plan_path),
            "--collection-targets-path",
            str(args.collection_targets_path),
        ]
        for gap in args.evaluation_gap or []:
            command.extend(["--evaluation-gap", str(gap)])
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "build-autobot-gap-materials":
        command = [
            sys.executable,
            "bot/gap_materials_builder.py",
            "--accepted-path",
            str(args.accepted_path),
            "--targets-path",
            str(args.targets_path),
            "--output-path",
            str(args.output_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        for request_id in args.blocked_request_id or []:
            command.extend(["--blocked-request-id", str(request_id)])
        for request_id in args.clear_blocked_request_id or []:
            command.extend(["--clear-blocked-request-id", str(request_id)])
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "enqueue-autobot-gap-curriculum":
        command = [
            sys.executable,
            "bot/enqueue_curriculum.py",
            "--curriculum-path",
            str(args.curriculum_path),
            "--queue-path",
            str(args.queue_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "run-autobot-gap-loop":
        command = [
            sys.executable,
            "bot/run_gap_loop.py",
            "--records-path",
            str(args.records_path),
            "--candidate-path",
            str(args.candidate_path),
            "--rejected-path",
            str(args.rejected_path),
            "--accepted-path",
            str(args.accepted_path),
            "--curriculum-path",
            str(args.curriculum_path),
            "--fixture-request-plan-path",
            str(args.fixture_request_plan_path),
            "--collection-targets-path",
            str(args.collection_targets_path),
            "--gap-output-path",
            str(args.gap_output_path),
            "--gap-curriculum-path",
            str(args.gap_curriculum_path),
            "--queue-path",
            str(args.queue_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        for gap in args.evaluation_gap or []:
            command.extend(["--evaluation-gap", str(gap)])
        for request_id in args.blocked_request_id or []:
            command.extend(["--blocked-request-id", str(request_id)])
        for request_id in args.clear_blocked_request_id or []:
            command.extend(["--clear-blocked-request-id", str(request_id)])
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-autobot-gap-loop-readiness":
        command = [
            sys.executable,
            "scripts/eval/autobot_gap_loop_readiness.py",
            "--loop-report-path",
            str(args.loop_report_path),
            "--collection-targets-path",
            str(args.collection_targets_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--isolation-audit-path",
            str(args.isolation_audit_path),
            "--min-accepted-count",
            str(args.min_accepted_count),
            "--min-gap-build-coverage",
            str(args.min_gap_build_coverage),
        ]
        if args.dataset_report_path:
            command.extend(["--dataset-report-path", str(args.dataset_report_path)])
        if args.gap_report_path:
            command.extend(["--gap-report-path", str(args.gap_report_path)])
        if args.enqueue_report_path:
            command.extend(["--enqueue-report-path", str(args.enqueue_report_path)])
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase7-isolation":
        command = [
            sys.executable,
            "scripts/eval/phase7_isolation_audit.py",
            "--train-path",
            str(args.train_path),
            "--evaluation-path",
            str(args.evaluation_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--max-signature-hamming-distance",
            str(args.max_signature_hamming_distance),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase7-completion":
        command = [
            sys.executable,
            "scripts/eval/phase7_completion_gate.py",
            "--readiness-path",
            str(args.readiness_path),
            "--isolation-path",
            str(args.isolation_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "apply-phase7-isolation-block-policy":
        command = [
            sys.executable,
            "scripts/eval/phase7_isolation_block_policy.py",
            "--audit-path",
            str(args.audit_path),
            "--targets-path",
            str(args.targets_path),
            "--report-path",
            str(args.report_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase8-completion":
        command = [
            sys.executable,
            "scripts/eval/phase8_completion_gate.py",
            "--comparison-path",
            str(args.comparison_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase8-evidence-cycle":
        command = [
            sys.executable,
            "scripts/eval/phase8_evidence_cycle.py",
            "--corpus",
            str(args.corpus),
            "--max-docs",
            str(args.max_docs),
            "--max-cases",
            str(args.max_cases),
            "--report-path",
            str(args.report_path),
        ]
        if args.pretrained_embedding_model:
            command.extend(["--pretrained-embedding-model", str(args.pretrained_embedding_model)])
        if args.cross_encoder_model:
            command.extend(["--cross-encoder-model", str(args.cross_encoder_model)])
        if args.no_history_update:
            command.append("--no-history-update")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "build-phase8-reference-request":
        command = [
            sys.executable,
            "scripts/eval/phase8_reference_collection_request.py",
            "--gate-path",
            str(args.gate_path),
            "--request-path",
            str(args.request_path),
            "--report-path",
            str(args.report_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-external-validity":
        command = [
            sys.executable,
            "scripts/eval/real_data_external_validity.py",
            "--corpus",
            str(args.corpus),
            "--max-docs",
            str(args.max_docs),
            "--max-cases",
            str(args.max_cases),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--history-path",
            str(args.history_path),
            "--regression-tolerance",
            str(args.regression_tolerance),
            "--pretrained-embedding-model",
            str(args.pretrained_embedding_model),
            "--cross-encoder-model",
            str(args.cross_encoder_model),
        ]
        if args.no_history_update:
            command.append("--no-history-update")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-external-validity-ladder":
        command = [
            sys.executable,
            "scripts/eval/real_data_external_validity_ladder.py",
            "--corpus",
            str(args.corpus),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--regression-tolerance",
            str(args.regression_tolerance),
        ]
        for profile in args.profile or []:
            command.extend(["--profile", str(profile)])
        if args.no_history_update:
            command.append("--no-history-update")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-ann-efficiency-roadmap":
        command = [
            sys.executable,
            "scripts/eval/ann_efficiency_roadmap_gate.py",
            "--energy-report-path",
            str(args.energy_report_path),
            "--external-validity-report-path",
            str(args.external_validity_report_path),
            "--external-ladder-report-path",
            str(args.external_ladder_report_path),
            "--energy-measurement-report-path",
            str(args.energy_measurement_report_path),
            "--operational-report-path",
            str(args.operational_report_path),
            "--output-report-path",
            str(args.output_report_path),
            "--output-summary-path",
            str(args.output_summary_path),
        ]
        if args.refresh_artifacts:
            command.append("--refresh-artifacts")
        if args.allow_missing_operational:
            command.append("--allow-missing-operational")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-sara-ann-comparison":
        command = [
            sys.executable,
            "scripts/eval/sara_ann_comparison_report.py",
            "--external-validity-report-path",
            str(args.external_validity_report_path),
            "--external-ladder-report-path",
            str(args.external_ladder_report_path),
            "--energy-measurement-report-path",
            str(args.energy_measurement_report_path),
            "--internal-maintenance-report-path",
            str(args.internal_maintenance_report_path),
            "--event-memory-report-path",
            str(args.event_memory_report_path),
            "--event-memory-maintenance-coupling-report-path",
            str(args.event_memory_maintenance_coupling_report_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-sparse-diffusion-block-readiness":
        command = [
            sys.executable,
            "scripts/eval/sparse_diffusion_block_readiness.py",
            "--block-count",
            str(args.block_count),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-rust-core-readiness":
        command = [
            sys.executable,
            "scripts/eval/rust_core_readiness.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        if args.run_cargo_test:
            command.append("--run-cargo-test")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-rust-core-benchmark":
        command = [
            sys.executable,
            "scripts/eval/rust_core_benchmark.py",
            "--iterations",
            str(args.iterations),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase10-completion":
        command = [
            sys.executable,
            "scripts/eval/phase10_completion_gate.py",
            "--readiness-path",
            str(args.readiness_path),
            "--benchmark-path",
            str(args.benchmark_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-research-benchmark-suite":
        command = [
            sys.executable,
            "scripts/eval/research_benchmark_suite.py",
            "--rust-iterations",
            str(args.rust_iterations),
            "--manifest-path",
            str(args.manifest_path),
            "--summary-path",
            str(args.summary_path),
        ]
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase9-completion":
        command = [
            sys.executable,
            "scripts/eval/phase9_completion_gate.py",
            "--manifest-path",
            str(args.manifest_path),
            "--protocol-path",
            str(args.protocol_path),
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-research-fixture-readiness":
        command = [
            sys.executable,
            "scripts/eval/research_fixture_readiness.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-neuromorphic-capability-matrix":
        command = [
            sys.executable,
            "scripts/eval/neuromorphic_capability_matrix.py",
            "--active-row-count",
            str(args.active_row_count),
            "--context-length",
            str(args.context_length),
            "--total-readout-size",
            str(args.total_readout_size),
            "--quantization-bits",
            str(args.quantization_bits),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        for profile in args.profile or []:
            command.extend(["--profile", str(profile)])
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase11-completion":
        command = [
            sys.executable,
            "scripts/eval/phase11_completion_gate.py",
            "--matrix-path",
            str(args.matrix_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-operator-dashboard":
        command = [
            sys.executable,
            "scripts/eval/operator_dashboard.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase12-completion":
        command = [
            sys.executable,
            "scripts/eval/phase12_completion_gate.py",
            "--dashboard-path",
            str(args.dashboard_path),
            "--guide-path",
            str(args.guide_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase13-completion":
        command = [
            sys.executable,
            "scripts/eval/phase13_completion_gate.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase14-completion":
        command = [
            sys.executable,
            "scripts/eval/phase14_completion_gate.py",
            "--benchmark-path",
            str(args.benchmark_path),
            "--manifest-path",
            str(args.manifest_path),
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase15-completion":
        command = [
            sys.executable,
            "scripts/eval/phase15_completion_gate.py",
            "--benchmark-path", str(args.benchmark_path),
            "--report-path", str(args.report_path),
            "--summary-path", str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase16-completion":
        command = [
            sys.executable,
            "scripts/eval/phase16_completion_gate.py",
            "--benchmark-path", str(args.benchmark_path),
            "--report-path", str(args.report_path),
            "--summary-path", str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase17-completion":
        command = [
            sys.executable,
            "scripts/eval/phase17_completion_gate.py",
            "--credit-path", str(args.credit_path),
            "--integration-path", str(args.integration_path),
            "--report-path", str(args.report_path),
            "--summary-path", str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase18-completion":
        command = [
            sys.executable,
            "scripts/eval/phase18_completion_gate.py",
            "--benchmark-path", str(args.benchmark_path),
            "--integration-path", str(args.integration_path),
            "--report-path", str(args.report_path),
            "--summary-path", str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase19-completion":
        command = [
            sys.executable,
            "scripts/eval/phase19_completion_gate.py",
            "--benchmark-path", str(args.benchmark_path),
            "--report-path", str(args.report_path),
            "--summary-path", str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-semantic-echo-field":
        command = [
            sys.executable, "scripts/eval/semantic_echo_field_benchmark.py",
            "--fixture-path", str(args.fixture_path), "--report-path", str(args.report_path),
            "--summary-path", str(args.summary_path), "--trace-path", str(args.trace_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase20-completion":
        command = [
            sys.executable, "scripts/eval/phase20_completion_gate.py",
            "--benchmark-path", str(args.benchmark_path), "--report-path", str(args.report_path),
            "--summary-path", str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-own-latent-learning":
        command = [
            sys.executable,
            "scripts/eval/own_latent_learning_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--history-path",
            str(args.history_path),
            "--train-sizes",
            str(args.train_sizes),
        ]
        if args.no_history_update:
            command.append("--no-history-update")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "build-own-latent-manifest":
        command = [
            sys.executable,
            "scripts/eval/own_latent_manifest_builder.py",
            "--materials-path",
            str(args.materials_path),
            "--manifest-path",
            str(args.manifest_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--width",
            str(args.width),
            "--max-events",
            str(args.max_events),
            "--max-terms",
            str(args.max_terms),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-dendritic-feedback-gate":
        command = [
            sys.executable,
            "scripts/eval/dendritic_feedback_gate_benchmark.py",
            "--event-budget",
            str(args.event_budget),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-sparse-plan-trace-verifier":
        command = [
            sys.executable,
            "scripts/eval/sparse_plan_trace_verifier.py",
            "--fixture-path",
            str(args.fixture_path),
            "--repair-path",
            str(args.repair_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-synesthetic-multimodal-binding":
        command = [
            sys.executable,
            "scripts/eval/synesthetic_multimodal_binding_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--cross-link-path",
            str(args.cross_link_path),
            "--binding-manifest-path",
            str(args.binding_manifest_path),
            "--latent-manifest-path",
            str(args.latent_manifest_path),
            "--trace-path",
            str(args.trace_path),
            "--plug-swap-path",
            str(args.plug_swap_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--window-ms",
            str(args.window_ms),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-sparse-reasoning-prior":
        command = [
            sys.executable,
            "scripts/eval/sparse_reasoning_prior_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--trace-path",
            str(args.trace_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-resonance-credit":
        command = [
            sys.executable,
            "scripts/eval/resonance_credit_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--trace-path",
            str(args.trace_path),
            "--state-path",
            str(args.state_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-adaptive-credit-field":
        command = [
            sys.executable,
            "scripts/eval/adaptive_credit_field_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--trace-path",
            str(args.trace_path),
            "--state-path",
            str(args.state_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-risa-structural-plasticity":
        command = [
            sys.executable,
            "scripts/eval/risa_structural_plasticity_benchmark.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-structural-interpolation":
        command = [
            sys.executable,
            "scripts/eval/structural_interpolation_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-structural-interpolation-external":
        command = [
            sys.executable,
            "scripts/eval/structural_interpolation_external_benchmark.py",
            "--manifest-path",
            str(args.manifest_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-structural-interpolation-event-memory":
        command = [
            sys.executable,
            "scripts/eval/structural_interpolation_event_memory_benchmark.py",
            "--manifest-path",
            str(args.manifest_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-next-level-structural":
        command = [
            sys.executable,
            "scripts/eval/next_level_structural_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-continual-horizon":
        command = [
            sys.executable,
            "scripts/eval/continual_horizon_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-continual-horizon-external":
        command = [
            sys.executable,
            "scripts/eval/continual_horizon_external_gate.py",
            "--manifest-path",
            str(args.manifest_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "build-continual-horizon-collection-request":
        command = [
            sys.executable,
            "scripts/eval/continual_horizon_collection_request.py",
            "--gate-path",
            str(args.gate_path),
            "--targets-path",
            str(args.targets_path),
            "--report-path",
            str(args.report_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase23-structural-fusion":
        command = [
            sys.executable,
            "scripts/eval/phase23_structural_fusion_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase23-external-multimodal":
        command = [
            sys.executable,
            "scripts/eval/phase23_external_multimodal_gate.py",
            "--manifest-path",
            str(args.manifest_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "build-phase23-multimodal-collection-request":
        command = [
            sys.executable,
            "scripts/eval/phase23_multimodal_collection_request.py",
            "--gate-path",
            str(args.gate_path),
            "--targets-path",
            str(args.targets_path),
            "--report-path",
            str(args.report_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase24-causal":
        command = [
            sys.executable,
            "scripts/eval/phase24_causal_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase25-agent-loop":
        command = [
            sys.executable,
            "scripts/eval/phase25_agent_loop_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-next-level-promotion-review":
        command = [
            sys.executable,
            "scripts/eval/next_level_promotion_review.py",
            "--evaluation-dir",
            str(args.evaluation_dir),
            "--report-path",
            str(args.report_path),
            "--gate-path",
            str(args.gate_path),
            "--journal-path",
            str(args.journal_path),
            "--approval-path",
            str(args.approval_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "record-next-level-human-approval":
        command = [
            sys.executable,
            "scripts/eval/next_level_human_approval.py",
            "--evaluation-dir",
            str(args.evaluation_dir),
            "--reviewer",
            str(args.reviewer),
            "--note",
            str(args.note),
            "--output-path",
            str(args.output_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-scale-up-readiness":
        command = [
            sys.executable,
            "scripts/eval/scale_up_experiment_readiness.py",
            "--promotion-gate",
            str(args.promotion_gate),
            "--external-gate",
            str(args.external_gate),
            "--multimodal-gate",
            str(args.multimodal_gate),
            "--preregistration-path",
            str(args.preregistration_path),
            "--output-path",
            str(args.output_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "register-scale-up-preregistration":
        command = [
            sys.executable,
            "scripts/eval/scale_up_preregistration.py",
            "--draft-path",
            str(args.draft_path),
            "--output-path",
            str(args.output_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase27-portable-runtime":
        command = [
            sys.executable,
            "scripts/eval/phase27_portable_runtime_readiness.py",
            "--output-path",
            str(args.output_path),
            "--rust-report-path",
            str(args.rust_report_path),
            "--tokenizer-report-path",
            str(args.tokenizer_report_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase27-tokenizer-acceleration":
        command = [
            sys.executable,
            "scripts/eval/phase27_tokenizer_acceleration_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--output-path",
            str(args.output_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase31-repetition-consolidation":
        command = [
            sys.executable,
            "scripts/eval/phase31_repetition_consolidation_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--output-path",
            str(args.output_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase31-repetition-reranking":
        command = [
            sys.executable,
            "scripts/eval/phase31_repetition_reranking_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--output-path",
            str(args.output_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "register-phase33-structured-edge-preregistration":
        command = [
            sys.executable,
            "scripts/eval/phase33_structured_edge_preregistration.py",
            "--draft-path",
            str(args.draft_path),
            "--output-path",
            str(args.output_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "build-phase33-structured-edge-preregistration-draft":
        command = [
            sys.executable,
            "scripts/eval/phase33_structured_edge_draft.py",
            "--fixture-path",
            str(args.fixture_path),
            "--draft-path",
            str(args.draft_path),
            "--environment-path",
            str(args.environment_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-level2-capability-matrix":
        command = [
            sys.executable,
            "scripts/eval/level2_capability_matrix.py",
            "--evaluation-dir",
            str(args.evaluation_dir),
            "--output-path",
            str(args.output_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-adaptive-credit-event-memory":
        command = [
            sys.executable,
            "scripts/eval/adaptive_credit_event_memory_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--trace-path",
            str(args.trace_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-resonance-credit-integration":
        command = [
            sys.executable,
            "scripts/eval/resonance_credit_integration_benchmark.py",
            "--trace-path",
            str(args.trace_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-event-state-cache":
        command = [
            sys.executable,
            "scripts/eval/event_state_cache_benchmark.py",
            "--fixture-path",
            str(args.fixture_path),
            "--candidate-path",
            str(args.candidate_path),
            "--manifest-path",
            str(args.manifest_path),
            "--trace-path",
            str(args.trace_path),
            "--state-path",
            str(args.state_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-event-state-cache-integration":
        command = [
            sys.executable,
            "scripts/eval/event_state_cache_integration_benchmark.py",
            "--manifest-path",
            str(args.manifest_path),
            "--trace-path",
            str(args.trace_path),
            "--round-trip-state-path",
            str(args.round_trip_state_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-architecture-migration":
        command = [
            sys.executable,
            "scripts/eval/architecture_migration_benchmark.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-architecture-migration-external":
        command = [
            sys.executable,
            "scripts/eval/architecture_migration_external_gate.py",
            "--manifest-path",
            str(args.manifest_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "build-architecture-migration-collection-request":
        result = subprocess.run([
            sys.executable, "scripts/eval/architecture_migration_collection_request.py",
            "--gate-path", str(args.gate_path), "--targets-path", str(args.targets_path),
            "--report-path", str(args.report_path),
        ])
        sys.exit(result.returncode)

    elif args.command == "build-architecture-migration-manifest":
        result = subprocess.run([
            sys.executable, "scripts/eval/architecture_migration_manifest_builder.py",
            "--input-path", str(args.input_path), "--output-path", str(args.output_path),
            "--report-path", str(args.report_path),
        ])
        sys.exit(result.returncode)

    elif args.command == "eval-architecture-migration-evidence-cycle":
        result = subprocess.run([sys.executable, "scripts/eval/architecture_migration_evidence_cycle.py", "--input-path", str(args.input_path), "--report-path", str(args.report_path)])
        sys.exit(result.returncode)

    elif args.command == "eval-event-memory-ingest-pipeline":
        command = [
            sys.executable,
            "scripts/eval/event_memory_ingest_pipeline.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-event-memory-maintenance-coupling":
        command = [
            sys.executable,
            "scripts/eval/event_memory_maintenance_coupling_benchmark.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-persistent-self-state":
        command = [
            sys.executable,
            "scripts/eval/persistent_self_state_benchmark.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-idle-replay":
        command = [
            sys.executable,
            "scripts/eval/idle_replay_benchmark.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-internal-maintenance-efficiency":
        command = [
            sys.executable,
            "scripts/eval/internal_maintenance_efficiency_benchmark.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-internal-practical-integration":
        command = [
            sys.executable,
            "scripts/eval/internal_practical_integration_benchmark.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-operator-llm-assistant-readiness":
        command = [
            sys.executable,
            "scripts/eval/operator_llm_assistant_readiness.py",
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        if args.enabled:
            command.append("--enabled")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-phase6-completion":
        result = subprocess.run([
            sys.executable,
            "scripts/eval/phase6_completion_gate.py",
            "--readiness-path",
            str(args.readiness_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ])
        sys.exit(result.returncode)

    elif args.command == "record-energy-measurement":
        command = [
            sys.executable,
            "scripts/eval/energy_measurement_readiness.py",
            "--append-measurement",
            "--measurement-path",
            str(args.measurement_path),
            "--run-id",
            str(args.run_id),
            "--system",
            str(args.system),
            "--task",
            str(args.task),
            "--success-count",
            str(args.success_count),
            "--joules",
            str(args.joules),
            "--source",
            str(args.source),
            "--session-id",
            str(args.session_id),
            "--protocol-version",
            str(args.protocol_version),
            "--pair-id",
            str(args.pair_id),
            "--replicate-index",
            str(args.replicate_index),
            "--environment-fingerprint",
            str(args.environment_fingerprint),
            "--task-fixture-hash",
            str(args.task_fixture_hash),
            "--success-criterion-id",
            str(args.success_criterion_id),
            "--measurement-boundary",
            str(args.measurement_boundary),
            "--measurement-tool",
            str(args.measurement_tool),
            "--cpu-model",
            str(args.cpu_model),
            "--thread-count",
            str(args.thread_count),
            "--process-affinity",
            str(args.process_affinity),
            "--power-mode",
            str(args.power_mode),
            "--warmup-count",
            str(args.warmup_count),
            "--measured-repetitions",
            str(args.measured_repetitions),
            "--trial-count",
            str(args.trial_count),
            "--run-order",
            str(args.run_order),
            "--max-success-rate-delta",
            str(args.max_success_rate_delta),
            "--min-paired-replicates-per-task",
            str(args.min_paired_replicates_per_task),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--session-plan-path",
            str(args.session_plan_path),
            "--session-plan-summary-path",
            str(args.session_plan_summary_path),
        ]
        if args.duration_seconds is not None:
            command.extend(["--duration-seconds", str(args.duration_seconds)])
        if args.average_watts is not None:
            command.extend(["--average-watts", str(args.average_watts)])
        if args.notes:
            command.extend(["--notes", str(args.notes)])
        if args.maintenance_selected_count is not None:
            command.extend(["--maintenance-selected-count", str(args.maintenance_selected_count)])
        if args.maintenance_phase_count is not None:
            command.extend(["--maintenance-phase-count", str(args.maintenance_phase_count)])
        if args.maintenance_refresh_count is not None:
            command.extend(["--maintenance-refresh-count", str(args.maintenance_refresh_count)])
        if args.maintenance_event_cost is not None:
            command.extend(["--maintenance-event-cost", str(args.maintenance_event_cost)])
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "run-physical-energy-pair":
        command = [
            sys.executable,
            "scripts/eval/physical_energy_pair_runner.py",
            "--pair-id",
            str(args.pair_id),
            "--replicate-index",
            str(args.replicate_index),
            "--corpus-path",
            str(args.corpus_path),
            "--max-docs",
            str(args.max_docs),
            "--max-cases",
            str(args.max_cases),
            "--repetitions",
            str(args.repetitions),
            "--warmup-count",
            str(args.warmup_count),
            "--thread-count",
            str(args.thread_count),
            "--process-affinity",
            str(args.process_affinity),
            "--power-mode",
            str(args.power_mode),
            "--measurement-tool",
            str(args.measurement_tool),
            "--sara-joules",
            str(args.sara_joules),
            "--ann-joules",
            str(args.ann_joules),
            "--measurement-path",
            str(args.measurement_path),
            "--meter-reading-path",
            str(args.meter_reading_path),
            "--manifest-path",
            str(args.manifest_path),
            "--trace-path",
            str(args.trace_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--meter-template-path",
            str(args.meter_template_path),
        ]
        if args.auto_system_energy_estimate:
            command.append("--auto-system-energy-estimate")
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "run-physical-energy-session-batch":
        command = [
            sys.executable,
            "scripts/eval/physical_energy_session_batch.py",
            "--session-plan-path",
            str(args.session_plan_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
        ]
        if args.execute_dry_run_pairs:
            command.append("--execute-dry-run-pairs")
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "eval-physical-energy-session-progress":
        command = [
            sys.executable,
            "scripts/eval/physical_energy_session_progress.py",
            "--batch-report-path",
            str(args.batch_report_path),
            "--measurement-path",
            str(args.measurement_path),
            "--report-path",
            str(args.report_path),
            "--summary-path",
            str(args.summary_path),
            "--internal-maintenance-report-path",
            str(args.internal_maintenance_report_path),
            "--event-memory-maintenance-coupling-report-path",
            str(args.event_memory_maintenance_coupling_report_path),
        ]
        result = subprocess.run(command)
        sys.exit(result.returncode)

    elif args.command == "clean":
        print("--- 環境のリセットを開始します ---")
        targets = ["data/interim", "data/processed"]
        for target in targets:
            if os.path.exists(target):
                for item in os.listdir(target):
                    if item == ".gitkeep": continue
                    path = os.path.join(target, item)
                    if os.path.isdir(path): shutil.rmtree(path)
                    else: os.remove(path)
                print(f"✅ {target} をクリーンアップしました。")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
