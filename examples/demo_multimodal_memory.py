# examples/demo_multimodal_memory.py
# マルチモーダル記憶・睡眠定着デモ
# 視覚・聴覚などの複数のモダリティからの入力をエンコードし、長期記憶(LTM)に保存し、睡眠プロセスを通じて記憶を定着させる一連のプロセスをデモンストレーションします。

import numpy as np
from sara_engine.encoders.vision import VisionEncoder
from sara_engine.encoders.audio import AudioEncoder
from sara_engine.memory.hippocampus import Hippocampus
from sara_engine.memory.ltm import LongTermMemory

def create_dummy_image():
    # 28x28のダミー画像データを生成
    return np.random.rand(28, 28)

def create_dummy_audio():
    # 1秒間のダミー音声データ (16kHz)
    return np.random.randn(16000)

def main():
    print("=== マルチモーダル記憶と睡眠による定着デモ ===")
    
    # モジュールの初期化
    print("モジュールを初期化中...")
    vision_encoder = VisionEncoder(input_shape=(28, 28), output_dim=64)
    audio_encoder = AudioEncoder(sample_rate=16000, output_dim=64)
    hippocampus = Hippocampus(memory_dim=128)
    ltm = LongTermMemory(capacity=1000, feature_dim=128)
    
    # 1. マルチモーダルデータのエンコード
    print("\n[フェーズ1: データのエンコード]")
    img = create_dummy_image()
    audio = create_dummy_audio()
    
    vision_features = vision_encoder.encode(img)
    audio_features = audio_encoder.encode(audio)
    
    print(f"視覚特徴量の形状: {vision_features.shape}")
    print(f"音声特徴量の形状: {audio_features.shape}")
    
    # 2. 短期記憶(海馬)への保存
    print("\n[フェーズ2: 海馬での短期記憶と連想]")
    # 視覚と音声の特徴を結合してエピソードとして記憶
    combined_features = np.concatenate([vision_features, audio_features])
    
    memory_id = hippocampus.store(combined_features)
    print(f"記憶を海馬に保存しました (ID: {memory_id})")
    
    # 連想テスト
    retrieved = hippocampus.retrieve(vision_features)
    print(f"視覚情報からの記憶検索結果: {retrieved is not None}")
    
    # 3. 睡眠による記憶の定着（LTMへの転送）
    print("\n[フェーズ3: 睡眠フェーズ（記憶の定着）]")
    print("睡眠プロセスを開始し、海馬からLTMへ記憶を転送します...")
    
    # 実際の海馬の実装に合わせて睡眠処理を呼び出し（簡略化）
    consolidated_count = 0
    memories_to_consolidate = hippocampus.get_memories_for_consolidation()
    for mem in memories_to_consolidate:
        ltm.store(mem)
        consolidated_count += 1
        
    hippocampus.clear_consolidated_memories()
    print(f"{consolidated_count} 件の記憶がLTMに定着しました。")
    print("海馬のリフレッシュが完了しました。")
    
    print("\nデモが完了しました。")

if __name__ == "__main__":
    main()