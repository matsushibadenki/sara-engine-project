_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_ltm_demo.py",
    "//": "タイトル: LTM動作検証デモ",
    "//": "目的: MEMORIZEとRECALLアクションが正常に機能し、永続化されているか確認する。"
}

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
from sara_engine import StatefulRLMAgent

def run_ltm_demo():
    print("=== Long Term Memory (LTM) Demo ===")
    
    # モデルパス（学習済みがあれば使う）
    model_path = "models/stateful_rl_trained.pkl"
    
    # 1. エージェント起動 (Session 1)
    print("\n[Session 1] Initializing Agent...")
    agent = StatefulRLMAgent(model_path=model_path)
    
    # LTMをクリア（テスト用）
    if agent.ltm:
        agent.ltm.clear()
        print("LTM cleared for demo.")
    
    # 情報を記憶させる
    secret_info = "The treasure is buried under the old oak tree."
    print(f"Action: Memorizing -> '{secret_info}'")
    response = agent.solve(f"MEMORIZE: {secret_info}", train_rl=False)
    print(f"Agent Response: {response}")
    
    # 2. エージェント終了 (Simulate Restart)
    print("\n[System] Shutting down agent...")
    del agent
    time.sleep(1)
    
    # 3. エージェント再起動 (Session 2)
    print("\n[Session 2] Restarting Agent (Loading LTM)...")
    agent_new = StatefulRLMAgent(model_path=model_path)
    
    # 情報を思い出させる（ドキュメントなしで検索）
    query = "RECALL: Where is the treasure?"
    print(f"Action: Recalling -> '{query}'")
    
    # 内部的に「treasure」のSDRとLTM内のSDRの類似度を計算
    answer = agent_new.solve(query, train_rl=False)
    
    print("\n" + "="*30)
    print(f"RECALLED INFO: {answer}")
    print("="*30)
    
    if "oak tree" in answer:
        print("SUCCESS: The agent remembered the information across sessions!")
    else:
        print("FAILURE: Could not recall.")

if __name__ == "__main__":
    run_ltm_demo()