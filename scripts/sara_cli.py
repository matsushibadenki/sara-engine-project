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

from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path
from data.collect_math import generate_math_corpus, default_math_database
from data.collect_docs import process_document
from eval.test_math_chat import run_math_chat
from eval.test_vision_inference import run_vision_inference
from scripts.utils.build_replay_data import build_replay_data, default_replay_output_path
from scripts.utils.memory_health import default_memory_health_report_path, inspect_inference_memory
from scripts.utils.upgrade_memory import (
    default_upgrade_report_path,
    default_upgraded_model_path,
    upgrade_inference_memory,
)
from scripts.utils.fix_memory import default_fix_report_path, default_fixed_model_path, fix_inference_memory
from scripts.utils.prune_memory import prune_model_memory
from scripts.utils.manage_db import SaraCorpusDB

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
