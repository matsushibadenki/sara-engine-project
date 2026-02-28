# ディレクトリパス: scripts/sara_cli.py
# ファイルの日本語タイトル: SARA統合コマンドラインインターフェース
# ファイルの目的や内容: 役割別に整理された各サブディレクトリのスクリプトを読み込み、一つのコマンドラインから呼び出せるようにする。

import argparse
import sys
import os

# scriptsディレクトリ自体をシステムパスに追加し、サブディレクトリをモジュールとして認識させる
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# サブディレクトリからのインポートに変更
from data.collect_math import generate_math_corpus, default_math_database
from data.collect_all import CorpusIntegrator
from data.collect_docs import process_document
from train.train_chat import train_chat_data
from eval.test_math_chat import run_math_chat

def main():
    parser = argparse.ArgumentParser(description="SARA Engine 統合管理CLI")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # 1. 数式コーパス生成コマンド
    parser_math = subparsers.add_parser("generate-math", help="数式のテキストとQ&Aコーパスを生成します。")
    parser_math.add_argument("--out_txt", default="data/interim/math_corpus.txt", help="テキストコーパスの出力先")
    parser_math.add_argument("--out_jsonl", default="data/interim/math_corpus.jsonl", help="Q&Aコーパスの出力先")

    # 2. ドキュメント抽出コマンド
    parser_docs = subparsers.add_parser("extract-docs", help="PDF, CSV, HTMLからテキストを抽出します。")
    parser_docs.add_argument("type", choices=["pdf", "csv", "html"], help="入力データの種類")
    parser_docs.add_argument("source", help="ファイルパスまたはURL")
    parser_docs.add_argument("--out_txt", default="data/interim/docs_corpus.txt", help="抽出テキストの出力先")

    # 3. コーパス統合コマンド
    parser_integrate = subparsers.add_parser("integrate-corpus", help="中間データを統合・重複排除して学習用コーパスを作成します。")
    parser_integrate.add_argument("--math_src", default="data/interim/math_corpus.txt", help="入力する数式コーパス")
    parser_integrate.add_argument("--docs_src", default="data/interim/docs_corpus.txt", help="入力するドキュメントコーパス")
    parser_integrate.add_argument("--out_corpus", default="data/processed/corpus.txt", help="統合後の出力先")

    # 4. 対話学習コマンド
    parser_train = subparsers.add_parser("train-chat", help="チャットデータをSNNに学習させます。")
    parser_train.add_argument("--sources", nargs="+", default=["data/raw/chat_data.jsonl", "data/interim/math_corpus.jsonl"], help="学習元のJSONLファイル（複数可）")
    parser_train.add_argument("--model", default="models/distilled_sara_llm.msgpack", help="SNNモデルの保存先")

    # 5. 対話テストコマンド
    parser_chat = subparsers.add_parser("chat", help="学習したSNNモデルと対話テストを行います。")
    parser_chat.add_argument("--model", default="models/distilled_sara_llm.msgpack", help="読み込むSNNモデル")

    args = parser.parse_args()

    if args.command == "generate-math":
        generate_math_corpus(default_math_database, args.out_txt, args.out_jsonl)
    elif args.command == "extract-docs":
        process_document(args.type, args.source, args.out_txt)
    elif args.command == "integrate-corpus":
        integrator = CorpusIntegrator(output_path=args.out_corpus)
        
        if os.path.exists(args.math_src):
            with open(args.math_src, "r", encoding="utf-8") as f:
                integrator.add_source(f.read(), source_type="math")
                
        if os.path.exists(args.docs_src):
            with open(args.docs_src, "r", encoding="utf-8") as f:
                integrator.add_source(f.read(), source_type="document")
                
    elif args.command == "train-chat":
        train_chat_data(args.sources, args.model)
    elif args.command == "chat":
        run_math_chat(args.model)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()