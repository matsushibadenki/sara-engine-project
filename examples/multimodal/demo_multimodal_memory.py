_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_multimodal_memory.py",
    "//": "タイトル: マルチモーダル記憶と睡眠による定着デモ",
    "//": "目的: site-packagesの古いモジュールを回避し、ローカルの最新モジュールから正しく読み込むようにパスとインポートを修正する。"
}

import os
import sys
import time

# ローカルのプロジェクトルートをパスの先頭に追加し、site-packagesより優先させる
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print("=== マルチモーダル記憶と睡眠による定着デモ ===")
    print("モジュールを初期化中...")
    
    try:
        # src配下から明示的にインポートするように修正
        from src.sara_engine.agent.sara_agent import SaraAgent
        from src.sara_engine.core.cortex import CorticalColumn
    except ImportError as e:
        print(f"モジュールの読み込みに失敗しました: {e}")
        return

    agent = SaraAgent()
    print("エージェントの初期化が完了しました。\n")

    print("--- 1. 視覚情報の入力 ---")
    vision_result = agent.perceive_image([0.9, 0.1, 0.1, 0.8], "猫")
    print(vision_result)
    time.sleep(1)

    print("\n--- 2. テキストによる対話（短期記憶の形成） ---")
    chat_result = agent.chat("さっき見たものは何ですか？", teaching_mode=False)
    print(f"SARA: {chat_result}")
    time.sleep(1)

    print("\n--- 3. 睡眠フェーズ（短期記憶から長期記憶への定着） ---")
    sleep_result = agent.sleep(consolidation_epochs=3)
    print(sleep_result)
    
    print("\nデモが完了しました。")

if __name__ == "__main__":
    main()