# {
#     "//": "ディレクトリパス: scripts/sara_cli.py",
#     "//": "ファイルの日本語タイトル: SARA統合コマンドラインインターフェース",
#     "//": "ファイルの目的や内容: データ収集、DB管理、そして【自己組織化学習】と【蒸留学習】の切り替えを一元管理する統合CLI。"
# }

import argparse
import sys
import os
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

from sara_engine.utils.project_paths import model_path
from data.collect_math import generate_math_corpus, default_math_database
from data.collect_docs import process_document
from eval.test_math_chat import run_math_chat
from eval.test_vision_inference import run_vision_inference
from scripts.utils.prune_memory import prune_model_memory
from scripts.utils.manage_db import SaraCorpusDB

def main():
    parser = argparse.ArgumentParser(description="SARA Engine 統合管理CLI - Data & Learning Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # --- 1. データ登録・管理 (Database) ---
    parser_db_import = subparsers.add_parser("db-import", help="テキスト(.txt)や対話データ(.jsonl)をDBに取り込みます。")
    parser_db_import.add_argument("file", help="取り込むファイルパス")

    parser_db_status = subparsers.add_parser("db-status", help="現在のコーパスDBの登録件数を表示します。")
    
    parser_db_export = subparsers.add_parser("db-export", help="DBから自己組織化用(TXT)と蒸留用(JSONL)にデータを一括出力します。")

    parser_db_reset = subparsers.add_parser("db-reset", help="コーパスDBを完全に初期化(空に)します。")

    # --- 2. 学習の実行 (Training) ---
    parser_train_self = subparsers.add_parser("train-self-org", help="【推奨】SNN固有の自己組織化学習(誤差逆伝播なし)を実行します。")
    
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

    parser_clean = subparsers.add_parser("clean", help="中間データを削除して環境をリセットします。")

    args = parser.parse_args()
    db_path = "data/sara_corpus.db"

    if args.command == "db-import":
        db = SaraCorpusDB(db_path)
        print(f"[DB] {args.file} をインポートしています...")
        added = db.import_file(args.file)
        print(f"✅ {added} 件のデータを新しくDBに登録しました。")

    elif args.command == "db-status":
        if not os.path.exists(db_path):
            print("DBが存在しません。まだデータが登録されていません。")
        else:
            db = SaraCorpusDB(db_path)
            stats = db.get_stats()
            print("=== SARA Corpus Database Status ===")
            total = 0
            for t_type, count in stats:
                print(f"- {t_type.capitalize()} データ: {count} 件")
                total += count
            print(f"合計: {total} 件")

    elif args.command == "db-export":
        db = SaraCorpusDB(db_path)
        print("[DB] 自己組織化学習用コーパス(corpus.txt)を出力しています...")
        c_count = db.export_for_self_organized("data/processed/corpus.txt")
        print(f"  -> {c_count} 件エクスポート完了")
        
        print("[DB] 蒸留学習用データ(chat_data.jsonl)を出力しています...")
        d_count = db.export_for_distillation("data/raw/chat_data.jsonl")
        print(f"  -> {d_count} 件エクスポート完了")
        print("✅ エクスポートが完了しました。学習を開始できます。")

    elif args.command == "db-reset":
        if os.path.exists(db_path):
            os.remove(db_path)
            print("🗑️ データベースを初期化しました。")
        else:
            print("データベースは既に空です。")

    elif args.command == "train-self-org":
        print("🧠 自己組織化学習(Self-Organized SNN)を開始します...")
        subprocess.run([sys.executable, "scripts/train/train_self_organized.py"])

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
