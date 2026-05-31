_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_feature_extraction.py",
    "//": "ファイルの日本語タイトル: SNN特徴抽出デモ",
    "//": "ファイルの目的や内容: SNNを用いてテキストをベクトル化し、行列演算(NumPy)を使わずに文章間の類似度を計算して比較する。"
}

import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.auto import AutoSNNModelForFeatureExtraction, AutoTokenizer
from sara_engine.pipelines import pipeline

def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    行列演算ライブラリ(NumPyなど)を使わず、標準の算術演算のみで
    コサイン類似度を計算する（制約準拠）。
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def main():
    print("=== SARA Engine: SNN Feature Extraction Demonstration ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "snn_feat_demo"))
    os.makedirs(workspace_dir, exist_ok=True)
    model_dir = os.path.join(workspace_dir, "snn_feat_checkpoint")
    
    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoSNNModelForFeatureExtraction.from_pretrained(model_dir)
    
    # Instantiate the feature-extraction pipeline
    extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    
    # 検索クエリ
    query = "誤差逆伝播法を使わない次世代のAIエンジン"
    
    # 比較対象のドキュメント群
    documents = [
        "誤差逆伝播法を排除した全く新しいAIエンジン",     # 意味・構文ともに近い (高類似度を期待)
        "行列演算に依存しないスパイク駆動のAIモデル",      # トピックは同じだが表現が違う (中程度を期待)
        "今日の天気はとても晴れていて絶好の散歩日和です",  # 全く無関係 (低類似度を期待)
    ]
    
    print(f"\n[Query]: {query}")
    print("-" * 50)
    
    # クエリのベクトル化
    query_vector = extractor(query)
    
    # 各ドキュメントとの類似度計算
    for i, doc in enumerate(documents):
        doc_vector = extractor(doc)
        similarity = compute_cosine_similarity(query_vector, doc_vector)
        
        print(f"Doc {i+1}: {doc}")
        print(f" -> Cosine Similarity: {similarity:.4f}\n")
        
    # SNNの状態を保存
    print(f"Saving model state to: {model_dir}")
    extractor.save_pretrained(model_dir)
    
    # ログ出力
    log_file = os.path.join(workspace_dir, "execution.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SARA SNN Feature Extraction Log ===\n")
        f.write("Status: SUCCESS\n")
        f.write(f"Model saved at: {model_dir}\n")
        f.write("Note: Vector similarities calculated without NumPy/Matrix Ops.\n")
        
    print(f"Execution log saved to {log_file}")
    print("=== Demonstration Completed ===")

if __name__ == "__main__":
    main()