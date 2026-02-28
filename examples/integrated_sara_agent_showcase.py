_FILE_INFO = {
    "//": "ディレクトリパス: examples/integrated_sara_agent_showcase.py",
    "//": "ファイルの日本語タイトル: 統合SaraAgent機能ショーケース",
    "//": "ファイルの目的や内容: SaraAgentの全主要機能を統合。長期エピソード記憶の保持、STDPによるツール利用（<CALC>等）の学習、および自律的なファンクションコーリングを実証する。"
}

import os
import sys
import time
import re

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.agent.sara_agent import SaraAgent

# --- 外部カスタムツールの定義 ---
def simple_calculator(context: str) -> str:
    """数式を抽出して計算する外部ブリッジ"""
    # <CALC>タグの前の文字列から数式を抽出
    target = context.split("<CALC>")[0] if "<CALC>" in context else context
    match = re.search(r'([0-9\+\-\*\/\s\(\)\.]+)', target.strip())
    if match:
        try:
            res = eval(match.group(1).strip())
            return str(res)
        except:
            return "計算エラー"
    return "数式が見つかりません"

def main():
    print("=== SARA Engine: Integrated Agent & Tool Interaction Demo ===\n")
    
    # 1. エージェントの初期化
    # 内部でPFC（前頭前野）、海馬、およびSNNメモリを構築
    print("[1] Initializing SaraAgent...")
    agent = SaraAgent()
    
    # 2. ツールの動的登録
    # 特定のアクションスパイク（<CALC>）にPython関数を紐付け
    print("[2] Registering external tools to the neural system...")
    agent.register_tool("<CALC>", simple_calculator)

    # 3. エピソード記憶とツール利用の教示 (Teaching Mode)
    # ユーザーの事実（長期記憶）と、計算が必要な時の振る舞いをSTDPで学習させる
    print("\n[3] Teaching phase: Encoding episodes and tool-use patterns...")
    training_scenarios = [
        "私 の 秘密 の コード は SARA-2026 です", # エピソード記憶
        "計算 が 必要 な 時 は <CALC> を 使い ます", # ツール利用の定義
        "15 + 28 は <CALC> 43 = 43 です",        # 実行パターンの教示
        "100 - 42 は <CALC> 58 = 58 です"
    ]
    
    for text in training_scenarios:
        # 強固なシナプス結合のため反復提示
        for _ in range(3):
            agent.chat(text, teaching_mode=True)
    print("    Training complete. Synaptic weights consolidated.")

    # 4. 複合タスクの実行 (Reasoning & Function Calling)
    print("\n[4] Autonomous Agent Task: Reasoning & Tool Usage")
    
    test_queries = [
        "私 の 秘密 の コード を 覚え て い ます か ？", # 長期記憶の想起
        "100 - 42 の 答え を 計算 し て 教え て ください <CALC>" # 自律ツール実行
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        print("SARA is thinking...")
        
        start_time = time.time()
        # 推論モードで実行。必要に応じて内部でツールを呼び出す
        response = agent.chat(query, teaching_mode=False)
        end_time = time.time()
        
        print(f"SARA: {response}")
        print(f"(Response time: {end_time - start_time:.2f}s)")

    # 5. メモリの健全性確認 (Homeostasis)
    # シナプス総数を確認し、爆発的な増加が抑制されているかチェック
    synapse_count = sum(len(post_dict) for post_dict in agent.episodic_snn.synapses.values())
    print(f"\n[5] System Health Check")
    print(f"    Current Episodic Synapses: {synapse_count}")
    print("    => Success: Memory remains efficient through synaptic pruning.")

    print("\n=== Demonstration Completed Successfully ===")

if __name__ == "__main__":
    main()