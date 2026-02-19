_FILE_INFO = {
    "//": "ディレクトリパス: examples/interactive_demo.py",
    "//": "タイトル: SARA Engine 統合デモ",
    "//": "目的: 仮想環境の古いパッケージを回避し、ローカルの最新のソースコードから正しく読み込むようにパスとインポートを修正する。"
}

import os
import sys

# プロジェクトルートをパスの先頭に追加し、site-packagesより優先させる
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# src配下から明示的にインポートするように修正
from src.sara_engine.models.rlm import StatefulRLMAgent

def run_demo():
    print("=== SARA Engine Interactive Demo ===")
    
    # モデル保存用ディレクトリ
    os.makedirs("models", exist_ok=True)
    model_path = "models/stateful_rl_trained.pkl"
    
    # エージェントの初期化（ローカルの最新コードを使用）
    agent = StatefulRLMAgent(model_path=model_path if os.path.exists(model_path) else None)
    
    print("\n[Step 1] Long Term Memory Test")
    # 情報を記憶させる
    memory_text = "The system password is ALPHA-CENTAURI."
    print(f"Action: Memorize -> '{memory_text}'")
    agent.solve(f"MEMORIZE: {memory_text}")
    
    print("\n[Step 2] Recall Test (Context-free)")
    # LTMから情報を検索
    query = "RECALL: What is the password?"
    response = agent.solve(query)
    print(f"Agent Recalled: {response}")
    
    print("\n[Step 3] Logical Reasoning Test")
    # コンテキストを与えて抽出
    doc = "Report: Today's mission code is RED-WOLF. Please keep it secret."
    query = "What is the mission code?"
    print(f"Document: {doc}")
    
    answer = agent.solve(query, document=doc)
    print(f"Agent Answer: {answer}")

if __name__ == "__main__":
    run_demo()