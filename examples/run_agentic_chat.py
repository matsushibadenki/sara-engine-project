_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_agentic_chat.py",
    "//": "タイトル: 自律型エージェントの対話デモ",
    "//": "目的: SaraAgentが文脈を自動判定し、破滅的忘却を防ぎながら学習・応答することを確認する。"
}

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.agent.sara_agent import SaraAgent

def run_demo():
    print("=== Sara Agent 起動中 (生物由来・省エネアーキテクチャ) ===\n")
    agent = SaraAgent()
    
    # --- 1. 学習フェーズ (Teaching) ---
    print("--- [学習フェーズ] ---")
    
    # Pythonに関する知識を入力
    msg1 = "python_expert: リスト内包表記を使うとPythonのコードは簡潔に書けます。"
    print(f"ユーザー: {msg1}")
    # 練度を上げるため3回学習させる
    for _ in range(3):
        res1 = agent.chat(msg1, teaching_mode=True)
    print(f"Agent: {res1}\n")
    
    # Rustに関する知識を入力（文脈が自動で切り替わるかテスト）
    msg2 = "rust_expert: Rustの所有権システムはメモリ安全性を保証します。"
    print(f"ユーザー: {msg2}")
    for _ in range(3):
        res2 = agent.chat(msg2, teaching_mode=True)
    print(f"Agent: {res2}\n")

    # --- 2. 想起フェーズ (Recall) ---
    print("--- [想起フェーズ] ---")
    
    # キーワードを少し変えて（ノイズを含めて）質問してみる
    q1 = "python_expert: リスト内包表記について教えて。"
    print(f"ユーザー: {q1}")
    print(f"Agent: {agent.chat(q1, teaching_mode=False)}\n")
    
    q2 = "rust_expert: メモリの安全性はどうやって守るの？"
    print(f"ユーザー: {q2}")
    print(f"Agent: {agent.chat(q2, teaching_mode=False)}\n")
    
    q3 = "biology: ミトコンドリアの役割は？"
    print(f"ユーザー: {q3}")
    print(f"Agent: {agent.chat(q3, teaching_mode=False)}\n")

if __name__ == "__main__":
    run_demo()