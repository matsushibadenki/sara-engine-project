_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_function_calling.py",
    "//": "ファイルの日本語タイトル: スパイキング・ファンクションコーリングのデモ (完全統合版エージェント対応)",
    "//": "ファイルの目的や内容: SNNが計算が必要な場面で<CALC>スパイクを発火させ、Python側のツールが計算結果を文脈に差し戻して回答を完成させる一連の流れをテストする。"
}

import os
from sara_engine.agent.sara_agent import SaraAgent

def main():
    print("===" * 15)
    print("[INFO] Starting Agentic SARA (Spiking Function Calling) Demo")
    print("===" * 15)
    
    # 完全版のSaraAgentは内部でモデルやボキャブラリーを自己構築するため、引数なしで初期化します
    print("[INFO] Initializing SaraAgent (with Retrieval-Augmented MoE and Hippocampus)...")
    agent = SaraAgent()

    # ---------------------------------------------------------
    # 1. 学習フェーズ (Teaching Mode)
    # SNNに「計算が必要な時は <CALC> スパイクを出し、= の後に回答を続ける」というシナプス結合を形成させる
    # ---------------------------------------------------------
    print("\n[TRAINING] Forming synapses for tool usage behavior...")
    
    # 教示データ
    training_data = [
        "calc 計算 ツール <CALC> = ",
        "15 + 28 は <CALC> 43 = 43 です。",
        "12 * 3 は <CALC> 36 = 36 です。",
        "100 - 42 は <CALC> 58 = 58 です。"
    ]
    
    # エージェントの教示モード(teaching_mode=True)を使ってエピソード記憶に焼き付ける
    for text in training_data:
        # 強固なシナプスを形成するため複数回反復学習
        for _ in range(3):
            agent.chat(text, teaching_mode=True)
            
    print("[TRAINING] Done.")

    # ---------------------------------------------------------
    # 2. 推論・エージェントフェーズ
    # 未知の計算式に対して、自律的にツールを呼び出せるかテスト
    # ---------------------------------------------------------
    prompt_text = "python_expert: 100 - 42 は <CALC>"
    
    print(f"\n[INFERENCE] Prompt: {prompt_text}")
    print("[INFERENCE] Agent takes over...")
    
    # SNNの推論とPythonブリッジの協調動作を開始
    response = agent.chat(prompt_text, teaching_mode=False)
    
    print("\n[RESULT] Final Output from Agent:")
    print(response)
    
    if "58 =" in response or "58" in response:
        print("\n[SUCCESS] SARA successfully fired the <CALC> action spike, received external input, and completed the thought process!")
    else:
        print("\n[FAILED] The function calling sequence was interrupted.")

if __name__ == "__main__":
    main()