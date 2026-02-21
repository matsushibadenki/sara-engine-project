_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_feature_extraction.py",
    "//": "ファイルの日本語タイトル: SNN特徴抽出パイプラインのデモ",
    "//": "ファイルの目的や内容: 恒常性シナプス可塑性(Homeostatic Plasticity)を適用し、助詞や句読点によるノイズ類似度を劇的に低下させるテスト。"
}

import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.models.spiking_feature_extractor import SpikingFeatureExtractor, SNNFeatureExtractorConfig
from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.pipelines import pipeline

def cosine_similarity(v1: list, v2: list) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 * norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def main():
    print("=== SNN Feature Extraction with Homeostatic Plasticity ===")
    print("Simulating Synaptic Downscaling to globally filter background noise.\n")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'feature_extraction_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 1. トークナイザーの準備
    tokenizer = SpikeTokenizer()
    texts_to_train = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleeping dog.",
        "I love programming in Python and Rust.",
        "人工知能は世界を変える技術です。",
        "AIは社会を大きく変化させるテクノロジーです。",
        "今日の夕食は美味しいカレーライスでした。"
    ]
    tokenizer.train(texts_to_train)
    tokenizer_path = os.path.join(workspace_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # 2. 特徴抽出モデルの準備 (Embedding次元を1024に拡大し波紋の衝突を抑える)
    config = SNNFeatureExtractorConfig(embedding_dim=1024, leak_rate=0.98, std_decay=0.2, std_recovery=0.05)
    model = SpikingFeatureExtractor(config)
    
    # 3. 恒常性シナプス可塑性 (Homeostatic Plasticity) によるノイズ抑制
    print("Habituating SNN with global corpus frequencies (Biological TF-IDF)...")
    tokenized_texts = [tokenizer.encode(t) for t in texts_to_train]
    model.habituate(tokenized_texts)
    
    model_dir = os.path.join(workspace_dir, "saved_feature_extractor")
    model.save_pretrained(model_dir)
    
    # 4. パイプラインの初期化
    print("Initializing Feature Extraction Pipeline...")
    extractor = pipeline("feature-extraction", model=model_dir, tokenizer=tokenizer)
    
    # 5. 類似度テスト
    print("\n--- Semantic Similarity Test (Cosine Similarity) ---")
    
    test_cases = [
        # 英語の類似ペア
        ("The quick brown fox jumps over the lazy dog.", "A fast brown fox leaps over a sleeping dog."),
        # 英語の非類似ペア
        ("The quick brown fox jumps over the lazy dog.", "I love programming in Python and Rust."),
        # 日本語の類似ペア
        ("人工知能は世界を変える技術です。", "AIは社会を大きく変化させるテクノロジーです。"),
        # 日本語の非類似ペア (前回はノイズで 0.5691 もあった)
        ("人工知能は世界を変える技術です。", "今日の夕食は美味しいカレーライスでした。")
    ]
    
    for text1, text2 in test_cases:
        vec1 = extractor(text1)
        vec2 = extractor(text2)
        sim = cosine_similarity(vec1, vec2)
        
        print(f"\nText A: '{text1}'")
        print(f"Text B: '{text2}'")
        print(f"-> Cosine Similarity: {sim:.4f}")

if __name__ == "__main__":
    main()