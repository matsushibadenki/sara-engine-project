# ディレクトリパス: scripts/sara_cli.py
# ファイルの日本語タイトル: SARA統合コマンドラインインターフェース
# ファイルの目的や内容: データ収集、統合、学習、推論テスト、そして記憶の刈り込み（プルーニング）などの処理を一元管理する。

import argparse
import sys
import os
import shutil

# scriptsディレクトリ自体をシステムパスに追加し、サブディレクトリをモジュールとして認識させる
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.collect_math import generate_math_corpus, default_math_database
from data.collect_all import CorpusIntegrator
from data.collect_docs import process_document
from train.train_chat import train_chat_data
from train.train_vision import train_vision_association
from eval.test_math_chat import run_math_chat
from eval.test_vision_inference import run_vision_inference
from utils.prune_memory import prune_model_memory  # 💡 新規追加

def main():
    parser = argparse.ArgumentParser(description="SARA Engine 統合管理CLI - Professional Edition")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # 1. 数式コーパス生成
    parser_math = subparsers.add_parser("generate-math", help="数式のテキストとQ&Aコーパスを生成します。")
    parser_math.add_argument("--out_txt", default="data/interim/math_corpus.txt")
    parser_math.add_argument("--out_jsonl", default="data/interim/math_corpus.jsonl")

    # 2. ドキュメント抽出
    parser_docs = subparsers.add_parser("extract-docs", help="多様なドキュメントからテキストを抽出します。")
    parser_docs.add_argument("type", choices=["pdf", "csv", "html"], help="ファイル形式")
    parser_docs.add_argument("source", help="パスまたはURL")
    parser_docs.add_argument("--out_txt", default="data/interim/docs_corpus.txt")

    # 3. コーパス統合
    parser_integrate = subparsers.add_parser("integrate-corpus", help="interim内の全データを統合して高品質な学習用コーパスを作成します。")
    parser_integrate.add_argument("--out_corpus", default="data/processed/corpus.txt", help="出力先")
    parser_integrate.add_argument("--dir", default="data/interim", help="スキャン対象のディレクトリ")

    # 4. 対話学習 (テキスト)
    parser_train = subparsers.add_parser("train-chat", help="チャット/数式データをSNNに蒸留学習させます。")
    parser_train.add_argument("--sources", nargs="+", default=["data/raw/chat_data.jsonl", "data/interim/math_corpus.jsonl"])
    parser_train.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    # 5. 対話テスト (チャットUI風)
    parser_chat = subparsers.add_parser("chat", help="学習済みSNNモデルと対話を行います。")
    parser_chat.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    # 6. 視覚連想学習
    parser_vtrain = subparsers.add_parser("train-vision", help="画像とテキストのペアを連想記憶として学習します。")
    parser_vtrain.add_argument("--csv", default="data/raw/visual/text/captions.csv")
    parser_vtrain.add_argument("--img_dir", default="data/raw/visual/images")
    parser_vtrain.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    # 7. 視覚推論テスト
    parser_vtest = subparsers.add_parser("vision-test", help="画像からSARAの連想（認識）を確認します。")
    parser_vtest.add_argument("image", help="テスト画像パス")
    parser_vtest.add_argument("--model", default="models/distilled_sara_llm.msgpack")

    # 8. 記憶の刈り込み (新規追加)
    parser_prune = subparsers.add_parser("prune", help="重みの低い不要な記憶を削除し、モデルを軽量化します。")
    parser_prune.add_argument("--model", default="models/distilled_sara_llm.msgpack", help="対象のモデルファイル")
    parser_prune.add_argument("--threshold", type=float, default=50.0, help="削除する重みの閾値（デフォルト: 50.0）")

    # 9. クリーンアップコマンド
    parser_clean = subparsers.add_parser("clean", help="中間データやキャッシュを削除して環境をリセットします。")
    parser_clean.add_argument("--all", action="store_true", help="processedデータもすべて削除します。")

    args = parser.parse_args()

    if args.command == "generate-math":
        generate_math_corpus(default_math_database, args.out_txt, args.out_jsonl)
    elif args.command == "extract-docs":
        process_document(args.type, args.source, args.out_txt)
    elif args.command == "integrate-corpus":
        print(f"--- コーパス統合を開始します ({args.dir} -> {args.out_corpus}) ---")
        integrator = CorpusIntegrator(output_path=args.out_corpus)
        if os.path.exists(args.dir):
            files = [f for f in os.listdir(args.dir) if f.endswith(".txt")]
            for filename in sorted(files):
                file_path = os.path.join(args.dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    source_type = "math" if "math" in filename else "document"
                    integrator.add_source(content, source_type=source_type)
        else:
            print(f"❌ ディレクトリが見つかりません: {args.dir}")
    elif args.command == "train-chat":
        train_chat_data(args.sources, args.model)
    elif args.command == "chat":
        run_math_chat(args.model)
    elif args.command == "train-vision":
        train_vision_association(args.csv, args.img_dir, args.model)
    elif args.command == "vision-test":
        run_vision_inference(args.image, args.model)
    elif args.command == "prune":
        prune_model_memory(args.model, args.threshold)
    elif args.command == "clean":
        print("--- 環境のリセットを開始します ---")
        targets = ["data/interim"]
        if args.all:
            targets.append("data/processed")
        for target in targets:
            if os.path.exists(target):
                for item in os.listdir(target):
                    if item == ".gitkeep": continue
                    path = os.path.join(target, item)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                print(f"✅ {target} をクリーンアップしました。")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()