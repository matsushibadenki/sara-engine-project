_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_rlm_demo.py",
    "//": "タイトル: RLM エージェント実行デモ",
    "//": "目的: 大規模テキストからの情報抽出タスクのデモ。"
}

import time
from sara_engine import StatefulRLMAgent

def main():
    print("=== SARA Engine - RLM Reasoning Demo ===")
    
    agent = StatefulRLMAgent()
    
    # ターゲットドキュメント
    document = (
        "CONFIDENTIAL LOG: Sector 7 is secure. "
        "The technician reported that the override code is BLUE-OCEAN-42. "
        "This code expires in 24 hours. "
        "System status: Nominal. End of log."
    )
    
    print(f"Document Length: {len(document)} characters")
    
    queries = [
        "What is the override code?",
        "Where is sector 7?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        start = time.time()
        answer = agent.solve(q, document, train_rl=False)
        print(f"Answer: {answer} (Time: {time.time()-start:.2f}s)")

if __name__ == "__main__":
    main()