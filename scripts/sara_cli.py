# ディレクトリパス: scripts/sara_cli.py
# ファイルの日本語タイトル: SARA統合コマンドラインインターフェース
# ファイルの目的や内容: 視覚推論テスト（vision-test）コマンドを追加し、画像からの連想を確認できるようにする。

import argparse
import sys
import os

# scriptsディレクトリ自体をシステムパスに追加し、サブディレクトリをモジュールとして認識させる
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.collect_math import generate_math_corpus, default_math_database
from data.collect_all import CorpusIntegrator
from data.collect_docs import process_document
from train.train_chat import train_chat_data
from train.train_vision import train_vision_association
from eval.test_math_chat import run_math_chat
from eval.test_vision_inference import run_vision_inference # 新規追加

def main():
    parser = argparse.ArgumentParser(description="SARA Engine 統合管理CLI")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # 1. 数式コーパス生成
    parser_math = subparsers.add_parser("generate-math", help="数式のテキストとQ&Aコーパスを生成します。")
    parser_math.add_argument("--out_txt", default="data/interim/math_corpus.txt")
    parser_math.add_argument("--out_jsonl", default="data/interim/math_corpus.jsonl")

    # 2. ドキュメント抽出
    parser_docs = subparsers.add_parser("extract-docs", help="PDF, CSV, HTMLからテキストを抽出します。")
    parser_docs.add_argument("type", choices=["pdf", "csv", "html"])
    parser_docs.add_argument("source")
    parser_docs.add_argument("--out_txt", default="data/interim/docs_corpus.txt")

    # 3. コーパス統合
    parser_integrate = subparsers.add_parser("integrate-corpus", help="中間データを統合して学習用コーパスを作成します。")
    parser_integrate.add_argument("--out_corpus", default="data/processed/corpus.txt")

    # 4. 対話学習 (テキスト)
    parser_train = subparsers.add_parser("train-chat", help="チャットデータをSNNに学習させます。")
    parser_train.add_argument("--sources", nargs="+", default=["data/raw/chat_data.jsonl", "data/interim/math_corpus.jsonl"])
    parser_train.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    # 5. 対話テスト (テキスト)
    parser_chat = subparsers.add_parser("chat", help="学習したSNNモデルと対話テストを行います。")
    parser_chat.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    # 6. 視覚連想学習
    parser_vtrain = subparsers.add_parser("train-vision", help="画像とテキストの連想学習を行います。")
    parser_vtrain.add_argument("--csv", default="data/raw/visual/text/captions.csv")
    parser_vtrain.add_argument("--img_dir", default="data/raw/visual/images")
    parser_vtrain.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    # 7. 視覚推論テスト (新規追加)
    parser_vtest = subparsers.add_parser("vision-test", help="画像を入力してSARAの連想（認識）を確認します。")
    parser_vtest.add_argument("image", help="テストする画像のパス")
    parser_vtest.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    args = parser.parse_args()

    if args.command == "generate-math":
        generate_math_corpus(default_math_database, args.out_txt, args.out_jsonl)
    elif args.command == "extract-docs":
        process_document(args.type, args.source, args.out_txt)
    elif args.command == "integrate-corpus":
        integrator = CorpusIntegrator(output_path=args.out_corpus)
        # 内部で interim の各ファイルを統合
        for src in ["data/interim/math_corpus.txt", "data/interim/docs_corpus.txt"]:
            if os.path.exists(src):
                with open(src, "r", encoding="utf-8") as f:
                    stype = "math" if "math" in src else "document"
                    integrator.add_source(f.read(), source_type=stype)
    elif args.command == "train-chat":
        train_chat_data(args.sources, args.model)
    elif args.command == "chat":
        run_math_chat(args.model)
    elif args.command == "train-vision":
        train_vision_association(args.csv, args.img_dir, args.model)
    elif args.command == "vision-test":
        run_vision_inference(args.image, args.model)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()