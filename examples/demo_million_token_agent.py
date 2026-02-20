# path: examples/demo_million_token_agent.py
# title: 100万トークン対応SaraAgentのエピソード記憶デモ
# purpose: 統合されたDynamicSNNMemoryを用いて、長大な会話文脈から直感的な連想記憶が機能するかをテストする。

import sys
import os

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sara_engine.agent.sara_agent import SaraAgent

def main():
    print("=== 100万トークン対応 SaraAgent エピソード記憶デモ ===")
    
    # 行列演算やGPUを使わず、CPU上のイベント駆動で動作するエージェントを初期化
    agent = SaraAgent()
    
    print("\n--- 1. エピソード（事実）の記憶（学習モード） ---")
    print("ユーザーの個人的な事実をSNNの動的グラフ構造にSTDPで刻み込みます。")
    episodes = [
        "私 の 名前 は 太郎 です",
        "最近 Python の 勉強 を 始め ました",
        "一番 好きな 果物 は リンゴ です"
    ]
    
    for ep in episodes:
        print(f"\nUser: {ep}")
        response = agent.chat(ep, teaching_mode=True)
        print(f"Sara: {response}")
        
    print("\n--- 2. 無関係な会話（ノイズ）の入力 ---")
    print("記憶の後に別の話題を挟み、文脈が上書きされないか（忘却耐性）を確認します。")
    noises = [
        "今日 の 天気 は 晴れ です",
        "生物学 の ミトコンドリア について 教え て",
        "視覚 情報 から 画像 を 認識 し ます"
    ]
    for noise in noises:
        agent.chat(noise, teaching_mode=False)
    print("（ノイズとなる会話を数ターン実行しました）")
    
    print("\n--- 3. 直感連想によるエピソードの想起（推論モード） ---")
    print("過去の長大なコンテキストの底流から、関連するキーワードをSNNが無意識に引っ張り出せるかをテストします。")
    questions = [
        "私 の 名前 は 何 です か ？",
        "私 の 好きな 果物 は 何 か 覚え て い ます か ？",
        "最近 私 が 始め た 勉強 は 何 です か ？"
    ]
    
    for q in questions:
        print(f"\nUser: {q}")
        response = agent.chat(q, teaching_mode=False)
        print(f"Sara:\n{response}")

if __name__ == "__main__":
    main()