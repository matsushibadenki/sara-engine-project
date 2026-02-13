_FILE_INFO = {
    "//": "ディレクトリパス: examples/interactive_demo.py",
    "//": "タイトル: SARA Engine 統合デモ",
    "//": "目的: インストール済みのsara_engineパッケージを使用して、記憶とQAを試す。"
}

import os
from sara_engine import StatefulRLMAgent

def run_demo():
    print("=== SARA Engine v0.1.3 Interactive Demo ===")
    
    # 1. エージェントの初期化
    # モデルファイルがカレントディレクトリになければ、初期状態で起動します
    model_path = "stateful_rl_trained.pkl"
    agent = StatefulRLMAgent(model_path=model_path if os.path.exists(model_path) else None)
    
    print("\n[Step 1] Long Term Memory Test")
    # 情報を記憶させる
    memory_text = "The system password is ALPHA-CENTAURI."
    print(f"Action: Memorize -> '{memory_text}'")
    agent.solve(f"MEMORIZE: {memory_text}")
    
    print("\n[Step 2] Recall Test (Context-free)")
    # 文脈なしで質問し、LTMからリコールできるか確認
    query = "RECALL: What is the password?"
    response = agent.solve(query)
    print(f"Agent Recalled: {response}")
    
    print("\n[Step 3] Logical Reasoning Test")
    # ドキュメントを提示して、その場で抽出できるか確認
    doc = "Report: Today's mission code is RED-WOLF. Please keep it secret."
    query = "What is the mission code?"
    print(f"Document: {doc}")
    print(f"Query: {query}")
    
    answer = agent.solve(query, document=doc)
    print(f"Agent Answer: {answer}")

if __name__ == "__main__":
    run_demo()