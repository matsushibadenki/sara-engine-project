# Directory Path: examples/core/demo_h_jepa_real_data.py
# English Title: Demonstration of H-JEPA with Real Text Data
# Purpose and Content: 階層的スパイキングJEPA(H-JEPA)を実際のテキスト時系列データに適用し、文字レベルのスパイク表現における未来状態の予測と自己組織化（予測誤差の減少）の過程をシミュレーションするデモスクリプト。多言語（英語・日本語など）のテキスト系列をスパイクとしてエンコードし学習させます。

import sys
import os
import time
from typing import List, Dict

# SARA Engineのモジュールパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sara_engine.models.spiking_jepa import SpikingJEPA

class SimpleTextSpikeEncoder:
    """
    テキストをH-JEPAに入力可能なスパイクIDのリストに変換する簡易エンコーダ。
    多言語対応のため、文字のUnicodeコードポイントをベースにスパイクIDを生成します。
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.next_id = 0

    def encode(self, text: str) -> List[List[int]]:
        """文字列を1文字ずつスパイクIDのリスト（各ステップの入力表現）に変換"""
        spikes_seq = []
        for char in text:
            if char not in self.char_to_id:
                # 語彙サイズ内に収めるためのハッシュ化（簡易的）
                self.char_to_id[char] = hash(char) % self.vocab_size
            
            # 各文字を「その文字のID」と「その文字のUnicodeカテゴリ等に基づく修飾ID」の2スパイクで表現する擬似的な分散表現
            char_id = self.char_to_id[char]
            unicode_id = (ord(char) % 50) + self.vocab_size # 特徴表現用のオフセット
            
            spikes_seq.append([char_id, unicode_id])
        return spikes_seq

def run_real_data_demo():
    print("==================================================")
    print(" H-JEPA 実データ（多言語テキスト）適用デモ")
    print("==================================================\n")

    # 1. デモ用多言語データセットの準備
    # H-JEPAにパターンを学習させるため、少し繰り返しの要素を含むテキストを用意
    sample_texts = [
        "SARA Engine is an SNN based cognitive architecture.",
        "SARAエンジンはSNNベースの認知アーキテクチャです。",
        "Moteur SARA est une architecture cognitive basée sur SNN."
    ]
    
    combined_text = " ".join(sample_texts)
    print("【入力テキストデータ】")
    print(combined_text)
    print("-" * 50)

    # 2. エンコーダの初期化とデータのスパイク化
    vocab_size = 500
    encoder = SimpleTextSpikeEncoder(vocab_size=vocab_size)
    spikes_sequence = encoder.encode(combined_text)
    
    print(f"総ステップ数 (文字数): {len(spikes_sequence)}")

    # 3. H-JEPAモデルの初期化
    # 表現次元はエンコーダの出力範囲に合わせて余裕を持たせる
    embed_dim = vocab_size + 100 
    layer_configs = [
        {"embed_dim": embed_dim, "hidden_dim": embed_dim * 2}
    ]
    
    # 時間スケールの設定
    # low: 次の文字, medium: 3文字先, high: 5文字先 のコンテキストから予測
    time_scales = {"low": 1, "medium": 3, "high": 5}
    
    jepa_model = SpikingJEPA(
        layer_configs=layer_configs,
        ema_decay=0.99,
        learning_rate=0.15, # 学習を早めるために少し高めに設定
        time_scales=time_scales
    )
    
    print("\n【H-JEPA モデル設定】")
    print(f"履歴バッファ長 (History Size): {len(jepa_model.history)}")
    print(f"時間スケール設定: {jepa_model.time_scales}")
    print("学習を開始します...\n")

    # メトリクス記録用
    history_surprise_counts = []
    
    # 4. 時系列ストリーム処理（予測と学習のループ）
    # 目標(Target)は、1ステップ先の未来の状態(文字)とする
    for t in range(len(spikes_sequence) - 1):
        x_spikes = spikes_sequence[t]      # 現在のコンテキスト
        y_spikes = spikes_sequence[t + 1]  # 未来のターゲット（自己教師あり）
        
        # 潜在変数 z は今回はノイズや文脈タグとして空のまま、あるいは固定値として扱う
        z_spikes: List[int] = [] 
        
        # H-JEPAによるフォワードパス (予測とPredictive Codingによる局所学習)
        predicted_s_y, surprise_signal = jepa_model.forward(
            x_spikes=x_spikes, 
            y_spikes=y_spikes, 
            learning=True
        )

        surprise_spikes, _ = jepa_model.minimizer.compute_surprise_signal(predicted_s_y, y_spikes)
        surprise_count = sum(surprise_spikes)
        history_surprise_counts.append(surprise_count)
        
        # 進捗を10ステップごとに表示
        if t % 10 == 0 or t == len(spikes_sequence) - 2:
            current_char = combined_text[t]
            next_char = combined_text[t + 1]
            # 直近10ステップの平均Surpriseを計算
            recent_surprise = sum(history_surprise_counts[-10:]) / min(len(history_surprise_counts), 10)
            
            print(f"[Step {t:3d}] Context:'{current_char}' -> Target:'{next_char}' | "
                  f"Surprise (Error) Spikes: {surprise_count:2d} | "
                  f"Recent Avg Surprise: {recent_surprise:.1f}")
            
            # 予測表現の一部を表示
            if predicted_s_y:
                print(f"          => 予測スパイク発火数: {len(predicted_s_y)}")
            else:
                print(f"          => 予測スパイクなし (学習初期または未学習パターン)")

    print("\n==================================================")
    print(" 実データへの適用デモ完了")
    print("==================================================")
    
    # 結果の簡単な分析
    early_avg = sum(history_surprise_counts[:20]) / 20
    late_avg = sum(history_surprise_counts[-20:]) / 20
    print(f"\n【学習効果の確認】")
    print(f"初期20ステップの平均予測誤差(Surprise)スパイク数 : {early_avg:.2f}")
    print(f"終盤20ステップの平均予測誤差(Surprise)スパイク数 : {late_avg:.2f}")
    
    if late_avg < early_avg:
        print("=> 成功: 時系列データの進行に伴い、H-JEPAの予測モデルが自己組織化され、予測誤差が減少しました！")
    else:
        print("=> 予測誤差が減少していません。パターンの複雑さに対して学習率やエポック数の調整が必要です。")

if __name__ == "__main__":
    run_real_data_demo()
