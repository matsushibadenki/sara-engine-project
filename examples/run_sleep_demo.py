_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_sleep_demo.py",
    "//": "タイトル: 睡眠(Sleep)による記憶定着と整理のデモ",
    "//": "目的: STDPのリプレイとLTMの整理が自律的に行われることを確認する。"
}

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.agent.sara_agent import SaraAgent

def run_sleep_demo():
    print("=== Sara Agent 起動 (睡眠・記憶定着システム) ===\n")
    agent = SaraAgent()
    
    print("--- [日中: 学習フェーズ] ---")
    # 同じ概念だが言い回しが少し違う文を教える（日常の些細な記憶の蓄積）
    msg1 = "python_expert: リスト内包表記はfor文より簡潔に書けます。"
    msg2 = "python_expert: リスト内包表記を使うとコードが短くなります。"
    msg3 = "python_expert: リスト内包表記はPythonの便利な機能です。"
    
    print(f"ユーザー: {msg1}\nAgent: {agent.chat(msg1, teaching_mode=True)}")
    print(f"ユーザー: {msg2}\nAgent: {agent.chat(msg2, teaching_mode=True)}")
    print(f"ユーザー: {msg3}\nAgent: {agent.chat(msg3, teaching_mode=True)}")
    
    print("\n[現在の海馬(LTM)の記憶数]:", len(agent.brain.ltm.memories))
    
    print("\n--- [日中: 想起フェーズ (睡眠前)] ---")
    q1 = "python_expert: リスト内包表記のメリットは？"
    print(f"ユーザー: {q1}")
    print(f"Agent: {agent.chat(q1, teaching_mode=False)}")
    print("(※まだ練度が低いため、確信度が低くなっています)")
    
    print("\n--- [夜間: 睡眠フェーズ] ---")
    print("Agent: Zzz... (大脳皮質へリプレイ中...)")
    sleep_result = agent.sleep(consolidation_epochs=20)
    print(f"Agent: {sleep_result}")
    
    print("\n--- [翌朝: 想起フェーズ (睡眠後)] ---")
    print(f"ユーザー: {q1}")
    print(f"Agent: {agent.chat(q1, teaching_mode=False)}")
    print("(※シナプスが強化され、パターン補完が完璧に働くため確信度が跳ね上がります！)")

if __name__ == "__main__":
    run_sleep_demo()