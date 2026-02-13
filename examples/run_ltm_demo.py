_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_ltm_demo.py",
    "//": "タイトル: 長期記憶 (LTM) 検証デモ",
    "//": "目的: 記憶の永続化と検索をテストする。"
}

import time
from sara_engine import StatefulRLMAgent

def run_ltm_demo():
    print("=== Long Term Memory (LTM) Demo ===")
    
    # 1. 記憶の保存
    print("\n[Session 1] Storing information...")
    agent = StatefulRLMAgent()
    
    # 以前の記憶があればクリア
    if agent.ltm:
        agent.ltm.clear()
        
    secret = "The secret code is CRIMSON-SKY-99."
    agent.solve(f"MEMORIZE: {secret}")
    print(f"Saved to LTM: {secret}")
    
    del agent
    time.sleep(1)
    
    # 2. セッションを跨いだ検索
    print("\n[Session 2] Recalling information...")
    agent_new = StatefulRLMAgent()
    query = "RECALL: What is the secret code?"
    
    answer = agent_new.solve(query)
    print(f"\nSARA Recalled: {answer}")
    
    if "CRIMSON" in answer:
        print("\nSUCCESS: LTM recall works across sessions.")
    else:
        print("\nFAILURE: Recall failed.")

if __name__ == "__main__":
    run_ltm_demo()