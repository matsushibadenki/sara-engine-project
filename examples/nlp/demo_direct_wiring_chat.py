# examples/nlp/demo_direct_wiring_chat.py
# 日本語タイトル: 直接シナプス結線SNN インタラクティブ対話デモ
# 目的: 学習済みのSNNモデルを読み込み、ユーザーからの任意のプロンプトに対してリアルタイムにテキストを生成する。
# {
#     "//": "学習済みのJSONモデルをロードし、対話形式で推論機能をテストします。"
# }

import os
import sys

# プロジェクトルートのsrcをパスに追加してモジュールを読み込めるようにする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sara_engine.learning.direct_wiring import DirectWiringSNN

def main():
    # ワークスペースパスの設定
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../workspace/logs/direct_wiring'))
    model_path = os.path.join(workspace_dir, 'direct_wiring_snn_model.json')

    if not os.path.exists(model_path):
        print(f"[ERROR] モデルが見つかりません。先に demo_direct_wiring.py を実行してください: {model_path}")
        return

    print("[INFO] Loading trained Direct Wiring SNN model...")
    snn = DirectWiringSNN(context_window=10)
    snn.load_model(model_path)
    print(f"[INFO] Model loaded successfully. Vocabulary size: {len(snn.char_to_id)} characters.")
    print("[INFO] Type 'quit' or 'exit' to stop.")
    print("-" * 50)

    # 対話ループ
    while True:
        try:
            prompt = input("\nUser Prompt > ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue

            print("SARA Engine > ", end="", flush=True)
            # 生成処理の実行（プロンプトの続きを生成）
            output = snn.generate(prompt=prompt, max_new_tokens=60)
            
            # 入力プロンプト部分を緑色などで装飾せずシンプルに出力
            print(output)

        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break

if __name__ == "__main__":
    main()