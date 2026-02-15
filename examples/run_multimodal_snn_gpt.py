_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_multimodal_snn_gpt.py",
    "//": "タイトル: SNN-GPTとマルチモーダル統合デモ",
    "//": "目的: 画像入力のバインディングと、SNN-GPTによる自律的な文章生成を検証する。"
}

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.agent.sara_agent import SaraAgent

def run_demo():
    print("=== Sara Agent 起動中 (SNN-GPT + Multimodal) ===\n")
    
    for f in ["sara_multimodal_ltm.pkl", "sara_vocab.json"]:
        if os.path.exists(f): os.remove(f)
        
    agent = SaraAgent()
    
    # --- 1. マルチモーダル視覚入力 (Vision) ---
    print("--- [視覚知覚フェーズ] ---")
    # 赤い丸い物体を模したダミーの画像特徴量配列
    dummy_image_features = [0.9 if i % 10 == 0 else 0.1 for i in range(1000)]
    res = agent.perceive_image(dummy_image_features, "リンゴ")
    print(res)
    
    # --- 2. 言語のシーケンス学習 (Teaching) ---
    print("\n--- [言語シーケンス学習フェーズ] ---")
    texts = [
        "python_expert: リスト 内包 表記 を 使う と コード は 簡潔 に 書け ます",
        "biology: ミトコンドリア は 細胞 の エネルギー を 作り ます",
        "vision: リンゴ は 赤く て 美味しい 果物 です"
    ]
    for text in texts:
        print(f"教示: {text}")
        agent.chat(text, teaching_mode=True)
        # 反復による結合強化
        agent.chat(text, teaching_mode=True)

    # --- 3. SNN-GPT 文章生成フェーズ (Generation) ---
    print("\n--- [SNN-GPT 自己回帰生成フェーズ] ---")
    
    prompts = [
        "リスト 内包 表記 を 使う と",
        "ミトコンドリア は 細胞 の",
        "リンゴ は"
    ]
    
    for p in prompts:
        print(f"\nユーザー(プロンプト): {p}")
        # GPTが続きの言葉を生成
        res = agent.chat(p, teaching_mode=False)
        print(f"Agent:\n{res}")

if __name__ == "__main__":
    run_demo()