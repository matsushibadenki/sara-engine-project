# ディレクトリパス: scripts/train/train_chat.py
# ファイルの日本語タイトル: チャットモデル学習スクリプト
# ファイルの目的や内容: collect_docs.pyで自動生成された対話ペア(JSONL)を読み込み、エージェントのSNNに学習させる。学習後にモデルを永続化する。

from sara_engine.utils.project_paths import (
    model_path as models_path,
    raw_data_path,
    resolve_project_relative,
)
from sara_engine.agent.sara_agent import SaraAgent
import os
import json
import sys

# srcディレクトリをパスに追加してモジュールをインポートできるようにする
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'src')))


def train_chat_model(
    data_path=raw_data_path("chat_data.jsonl"),
    save_dir=models_path("sara_agent"),
    epochs=2,
):
    print(f"--- チャット学習開始: {data_path} ---")

    data_path = resolve_project_relative(data_path)

    if not os.path.exists(data_path):
        print(f"❌ 学習データが見つかりません: {data_path}")
        return

    agent = SaraAgent()

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pairs = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            prompt = data.get("prompt")
            response = data.get("response") or data.get("completion")
            if prompt and response:
                pairs.append((prompt, response))
        except json.JSONDecodeError:
            continue

    if not pairs:
        print("⚠️ 有効な対話ペアが見つかりませんでした。")
        return

    print(f"✅ {len(pairs)} 件の対話ペアを読み込みました。SNNに学習させます...")

    for epoch in range(epochs):
        print(f"\nエポック {epoch + 1}/{epochs}")
        for idx, (prompt, response) in enumerate(pairs):
            combined_text = f"general: 質問「{prompt}」に対する回答は「{response}」"
            agent.chat(combined_text, teaching_mode=True)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(pairs):
                print(f"  {idx + 1}/{len(pairs)} 件完了...")

    print("🎉 チャットモデルのSNN学習が完了しました。")

    print("💾 エージェントの学習状態をディスクに保存しています...")
    agent.save_agent(save_dir)
    print("✨ すべての保存処理が完了しました。")


def train_chat_data(data_paths=None, model_path=None, epochs=2):
    """Backward-compatible wrapper for older CLI callers."""
    if not data_paths:
        data_path = raw_data_path("chat_data.jsonl")
    elif isinstance(data_paths, list):
        data_path = data_paths[0]
    else:
        data_path = str(data_paths)
    save_dir = model_path or models_path("sara_agent")
    train_chat_model(data_path=data_path, save_dir=save_dir, epochs=epochs)


if __name__ == "__main__":
    train_chat_model()
