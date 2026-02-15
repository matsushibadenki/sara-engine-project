_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_agentic_chat.py",
    "//": "タイトル: 自律型エージェント(PFC+Hippocampus+RLM)の対話デモ",
    "//": "目的: SaraAgentが文脈を自動判定し、記憶の検索からRLMによる抽出までを一貫して行うことを確認する。"
}

import os
import sys

# プロジェクトルート付近のsrcディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.agent.sara_agent import SaraAgent

def run_demo():
    print("=== Sara Agent 起動中 (PFC + Hippocampus + RLM 統合アーキテクチャ) ===\n")
    
    # 既存の記憶ファイルをクリアしてクリーンな状態でテスト
    if os.path.exists("sara_agent_ltm.pkl"):
        os.remove("sara_agent_ltm.pkl")
    if os.path.exists("sara_vocab.json"):
        os.remove("sara_vocab.json")
        
    agent = SaraAgent()
    
    # --- 1. 学習フェーズ (Teaching) ---
    print("--- [学習フェーズ] ---")
    
    messages = [
        "python_expert: リスト 内包 表記 を 使う と Python の コード は 簡潔 に 書け ます",
        "rust_expert: Rust の 所有 権 システム は メモリ 安全 性 を 保証 します",
        "biology: ミトコンドリア は 細胞 の エネルギー を 作り ます"
    ]
    
    for msg in messages:
        print(f"ユーザー(教示): {msg}")
        # 確実に定着させるために2回経験させる
        agent.chat(msg, teaching_mode=True)
        res = agent.chat(msg, teaching_mode=True)
        print(f"Agent: {res}\n")

    # --- 2. 睡眠フェーズ (Consolidation) ---
    print("--- [睡眠フェーズ] ---")
    print(agent.sleep(consolidation_epochs=3))
    print("\n")

    # --- 3. 想起・推論フェーズ (Recall & RLM) ---
    print("--- [想起・推論フェーズ (ICL + RLM)] ---")
    
    questions = [
        "Python の コード を 簡潔 に 書く には ？",
        "メモリ 安全 性 は どう やって 守る の ？",
        "細胞 の エネルギー は 何 が 作る ？"
    ]
    
    for q in questions:
        print(f"ユーザー(質問): {q}")
        res = agent.chat(q, teaching_mode=False)
        print(f"Agent:\n{res}\n")
        
    # クリーンアップ
    if os.path.exists("sara_agent_ltm.pkl"):
        os.remove("sara_agent_ltm.pkl")
    if os.path.exists("sara_vocab.json"):
        os.remove("sara_vocab.json")

if __name__ == "__main__":
    run_demo()