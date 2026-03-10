# ファイルパス: scripts/data/collect_twitter.py
# ファイル名: collect_twitter.py
# ファイルの内容: コマンドラインからTwitter(X)の該当URLのツイートや画像を収集するための実行スクリプト。

import argparse
import sys

from sara_engine.utils.data.twitter import TwitterDataCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Twitter(X)からテキスト・画像データを収集するスクリプトです。"
    )
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        required=True,
        help="データ収集対象となるTwitterの公開URL（複数指定可能）。",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="保存時のラベル名（例: tech, news など）。保存ディレクトリ名にも使用されます。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="取得ごとの待機時間（秒）。アクセス過多防止用。デフォルト1.0秒。",
    )

    args = parser.parse_args()

    print(f"[{args.label}] のデータ収集を開始します (計 {len(args.urls)} 件のURL)...")
    try:
        collector = TwitterDataCollector()
        manifest_path = collector.download_tweets(
            urls=args.urls,
            label=args.label,
            sleep_seconds=args.sleep,
        )
        print("収集が完了しました！")
        print(f"保存先マニフェストファイル: {manifest_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
