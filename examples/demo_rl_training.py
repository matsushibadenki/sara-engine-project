# 配置するディレクトリのパス: examples/demo_rl_training.py
# ファイルの日本語タイトル: 強化学習（RL）モジュールの学習デモ
# ファイルの目的や内容: 仮想環境(site-packages)の古いモジュールではなく、ローカルの最新モジュールを読み込むようにパス追加の修正を行ったデモスクリプト。

import sys
import os

# プロジェクトルートをパスに追加し、ローカルのsrcを参照させる
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.models.rlm import StatefulRLMAgent

def main():
    print("=== RLM 強化学習デモンストレーション ===")
    
    print("Stateful RLM エージェントを初期化中...")
    try:
        agent = StatefulRLMAgent()
    except ImportError as e:
        print(f"モジュールの読み込みに失敗しました: {e}")
        return
        
    print("\n学習ループを開始します...")
    
    # ダミーの文書とクエリ
    document = "Pythonのリスト内包表記は、簡潔なコードを書くのに非常に便利です。また、Rustはメモリ安全性を保証します。"
    queries = ["Python コード 簡潔", "Rust メモリ"]
    
    num_episodes = 5
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        query = queries[episode % len(queries)]
        print(f"Query: {query}")
        
        # solveメソッド内部で、エージェントは自律的に状態遷移(READ/EXTRACT等)を行い、
        # train_rl=True により報酬に基づいたSTDP的な重み更新が実行されます
        found_info = agent.solve(query=query, document=document, train_rl=True)
        
        if found_info:
            print(f"  抽出結果: {found_info}")
        else:
            print("  情報の抽出に失敗しました (DONE状態到達)。")
            
        print(f"Episode {episode + 1} 完了")

    print("\n強化学習デモが完了しました。")

if __name__ == "__main__":
    main()