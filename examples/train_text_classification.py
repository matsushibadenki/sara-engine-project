# directory: examples/train_text_classification.py
# title: Text Classification Example with SARA Engine
# description: テキスト（単語/文字）をスパイク列に変換し、SARAエンジンでカテゴリ分類を行うサンプルコード。

import sys
import os
import numpy as np

# ライブラリのインポート（パス解決）
try:
    from sara_engine import SaraEngine
except ImportError:
    # ローカル開発用に親ディレクトリをパスに追加する場合
    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
    try:
        from sara_engine import SaraEngine
    except ImportError:
        print("Error: 'sara_engine' module not found. Please install it via pip or check your path.")
        sys.exit(1)

def text_to_spikes(text, vocab_map, steps_per_char=3):
    """
    テキストをスパイク列に変換する関数
    
    Args:
        text (str): 入力テキスト
        vocab_map (dict): 文字とニューロンIDの対応表
        steps_per_char (int): 1文字あたりの持続ステップ数（リザーバへの刺激を強めるため複数回発火させる）
    
    Returns:
        List[List[int]]: スパイク列 [[fired_neuron_idx, ...], [fired_neuron_idx, ...], ...]
    """
    spike_train = []
    
    for char in text:
        # 文字をIDに変換（未知の文字は無視または特定IDへ）
        if char in vocab_map:
            neuron_idx = vocab_map[char]
            
            # 1文字に対して数ステップ連続で入力（刺激を確実にするため）
            for _ in range(steps_per_char):
                spike_train.append([neuron_idx])
        else:
            # 空のステップ（ノイズまたは区切り）
            spike_train.append([])
            
    # 文末に少し「余韻（Echo）」を持たせるための空白ステップを追加
    # これにより、リザーバが文脈を処理する時間を確保する
    for _ in range(10):
        spike_train.append([])
        
    return spike_train

def main():
    # 1. データセットの準備（簡易的な感情分析: Positive / Negative）
    # シンプルな単語ベースの学習
    data = [
        ("good", 0), ("great", 0), ("happy", 0), ("excellent", 0), ("yes", 0),
        ("bad", 1), ("worst", 1), ("sad", 1), ("poor", 1), ("no", 1),
        ("nice", 0), ("awful", 1)
    ]
    
    # 語彙辞書の作成（a-zのみを対象とする簡易版）
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab_map = {c: i for i, c in enumerate(chars)}
    input_size = len(chars)  # 26文字
    output_size = 2          # 0: Positive, 1: Negative
    
    print("Initializing SARA Engine for Text...")
    # テキストは画像より情報密度が低い場合があるため、レイヤー設定はそのままで機能します
    engine = SaraEngine(input_size=input_size, output_size=output_size)
    
    epochs = 10
    print(f"Start Training (Epochs: {epochs})...")
    
    for epoch in range(epochs):
        # データをシャッフル
        indices = np.random.permutation(len(data))
        correct = 0
        
        for idx in indices:
            text, label = data[idx]
            
            # スパイク変換
            spike_train = text_to_spikes(text, vocab_map, steps_per_char=2)
            
            # 学習
            engine.train_step(spike_train, target_label=label)
            
            # 予測（学習データでの精度確認）
            pred = engine.predict(spike_train)
            if pred == label:
                correct += 1
        
        accuracy = correct / len(data) * 100
        print(f"Epoch {epoch+1}: Accuracy {accuracy:.1f}%")
        
        # 睡眠フェーズ（過学習抑制）
        engine.sleep_phase(prune_rate=0.02)

    # テスト
    test_words = ["happy", "bad", "great", "awful"]
    print("\n--- Test Results ---")
    for word in test_words:
        spikes = text_to_spikes(word, vocab_map)
        pred = engine.predict(spikes)
        result = "Positive" if pred == 0 else "Negative"
        print(f"Input: '{word}' -> Prediction: {result}")

if __name__ == "__main__":
    main()