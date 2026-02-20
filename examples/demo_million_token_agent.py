_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_million_token_agent.py",
    "//": "タイトル: 100万トークン対応SaraAgentの長時間対話耐久テスト",
    "//": "目的: DynamicSNNMemoryとSaraAgentを統合し、長時間のチャット（大量のノイズ入力）後もメモリが破綻せず、恒常性によって省メモリを維持したまま初期エピソードを正しく想起できるか検証する。"
}

import sys
import os
import time
import random

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sara_engine.agent.sara_agent import SaraAgent

def main():
    print("=== 100万トークン対応 SaraAgent 長時間対話耐久テスト ===")
    
    # 行列演算やGPUを使わず、CPU上のイベント駆動で動作するエージェントを初期化
    agent = SaraAgent()
    
    print("\n--- 1. 重要なエピソードの記憶（学習モード） ---")
    print("ユーザーの個人的な事実をSNNの動的グラフ構造にSTDPで刻み込みます。")
    episodes = [
        "私 の 秘密 の 暗証 番号 は 7777 です",
        "私 の 好き な 映画 は マトリックス です",
        "私 の 出身 地 は 埼玉 県 戸田 市 です"
    ]
    
    for ep in episodes:
        print(f"User: {ep}")
        response = agent.chat(ep, teaching_mode=True)
        print(f"Sara: {response}")
        
    print("\n--- 2. 長時間のノイズ対話（耐久テスト） ---")
    print("10,000ターンの無関係な対話をシミュレートし、SNNメモリに負荷をかけます...")
    print("※ シナプス刈り込みと恒常性により、メモリ爆発が起きないことを確認します。")
    
    start_time = time.time()
    
    # 対話をシミュレートするためのダミー語彙
    dummy_words = [
        "今日", "天気", "晴れ", "雨", "曇り", "ご飯", "美味しい", "仕事", "疲れ", "た", 
        "楽しい", "映画", "音楽", "本", "読書", "睡眠", "散歩", "走る", "泳ぐ", "勉強", 
        "Python", "Rust", "SNN", "AI", "猫", "犬", "鳥", "花", "山", "海", "りんご", "みかん"
    ]
    
    # 乱数シードを固定せず、毎回異なるノイズを生成する
    for i in range(1, 10001):
        # 3〜7単語からなるランダムな文を生成
        sentence_length = random.randint(3, 7)
        noise_sentence = " ".join(random.choices(dummy_words, k=sentence_length))
        
        # 記憶領域にノイズを蓄積させるため学習モードで入力
        agent.chat(noise_sentence, teaching_mode=True) 
        
        # 2000ターンごとにSNNメモリの健全性（シナプス総数）をレポート
        if i % 2000 == 0:
            synapse_count = sum(len(post_dict) for post_dict in agent.episodic_snn.synapses.values())
            elapsed = time.time() - start_time
            print(f" [Turn {i:>5}] 経過時間: {elapsed:>5.2f}秒 | SNNシナプス総数: {synapse_count}")

    print(f"\nノイズ対話10,000ターン完了。総経過時間: {time.time() - start_time:.2f}秒")
    
    print("\n--- 3. 直感連想によるエピソードの想起（推論モード） ---")
    print("10,000ターンのノイズ後も、初期の重要な事実が忘却されずに残っているか確認します。")
    questions = [
        "私 の 秘密 の 暗証 番号 は 何 です か ？",
        "私 の 好き な 映画 は 何 か 覚え て い ます か ？",
        "私 の 出身 地 は どこ です か ？"
    ]
    
    for q in questions:
        print(f"\nUser: {q}")
        # 推論モードで海馬とSNNから記憶を呼び出す
        response = agent.chat(q, teaching_mode=False)
        print(f"Sara:\n{response}")

if __name__ == "__main__":
    main()