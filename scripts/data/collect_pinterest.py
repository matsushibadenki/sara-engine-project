# ファイルパス: scripts/data/collect_pinterest.py
# ファイル名: collect_pinterest.py
# ファイルの内容: コマンドラインからPinterestの該当URLのピンや画像を収集するための実行スクリプト。

import argparse
import sys

from sara_engine.utils.data.pinterest import PinterestImageCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pinterestから画像データを収集するスクリプトです。"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="データ収集対象となるPinterestの公開URL（ボードやピンなど）。",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="保存時のラベル名（例: cats, dogs など）。保存ディレクトリ名にも使用されます。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="収集する最大画像数。デフォルトは50です。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="画像取得ごとの待機時間（秒）。アクセス過多防止用。デフォルト0.5秒。",
    )

    args = parser.parse_args()

    print(f"[{args.label}] の画像の収集を開始します (上限: {args.limit}枚)...")
    try:
        collector = PinterestImageCollector()
        manifest_path = collector.download_images(
            page_url=args.url,
            label=args.label,
            limit=args.limit,
            sleep_seconds=args.sleep,
        )
        print("収集が完了しました！")
        print(f"保存先マニフェストファイル: {manifest_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
