_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_stateful_rlm.py",
    "//": "タイトル: Stateful RLM 実行デモ",
    "//": "目的: クラス名を整合させた最新版の実行テスト。"
}

from sara_engine import StatefulRLMAgent

def run_test():
    print("=== Stateful RLM Test ===")
    agent = StatefulRLMAgent()
    
    doc = "Memo: The project name is SARA. The lead developer is MATSUSHIBADENKI."
    query = "Who is the lead developer?"
    
    print(f"Context: {doc}")
    print(f"Query: {query}")
    
    answer = agent.solve(query, doc)
    print(f"\nResult: {answer}")

if __name__ == "__main__":
    run_test()