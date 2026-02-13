_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_stateful_rlm.py",
    "//": "タイトル: Stateful RLM 実行デモ",
    "//": "目的: 学習済みモデルを使ってドキュメント検索タスクを解決する。"
}

import sys
import os


from sara_engine import StatefulRLMAgent

def run_rlm_demo():
    print("=== Stateful RLM Agent Demo ===")
    
    # 1. 学習済みモデルをロード
    model_path = "models/stateful_demo.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please run 'python examples/train_stateful_demo.py' first.")
        return

    agent = StatefulRLMAgent(model_path=model_path)
    
    # 2. タスク設定
    query = "What is the master code?"
    
    # ターゲットドキュメント（答えが埋もれている）
    document = (
        "Project SARA log entry 2024. The system status is nominal. "
        "Checking sector 1... No anomalies. Checking sector 2... All clear. "
        "Security protocols are active. "
        "Confidential Section: The master override code is BLUE-OCEAN-42. "
        "Please memorize this code. "
        "End of log."
    )
    
    print(f"\nDocument Length: {len(document)} chars")
    
    # 3. エージェント実行
    answer = agent.solve(query, document)
    
    print("\n" + "="*30)
    print(f"FINAL RESULT: {answer}")
    print("="*30)
    
    if "BLUE-OCEAN-42" in answer:
        print("SUCCESS: The agent found the correct code!")
    else:
        print("FAILURE: The agent failed to find the code.")

if __name__ == "__main__":
    run_rlm_demo()