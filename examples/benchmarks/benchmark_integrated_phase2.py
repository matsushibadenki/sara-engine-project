_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmarks/benchmark_integrated_phase2.py",
    "//": "ファイルの日本語タイトル: フェーズ2新機能 統合ベンチマーク",
    "//": "ファイルの目的や内容: 新規実装した HierarchicalSDRPooling, SDRFuzzyAttention, SpikingMultimodalHub の動作を検証する。行列演算を使わずに特徴抽出とマルチモーダル連想ができるかテストする。"
}

import os
import sys
import time
import random

# ローカルのsrcを優先インポート
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from sara_engine.nn.pooling import HierarchicalSDRPooling
from sara_engine.nn.attention import SDRFuzzyAttention
from sara_engine.models.spiking_multimodal_hub import SpikingMultimodalHub

def run_benchmarks():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "workspace", "logs"))
    os.makedirs(workspace_dir, exist_ok=True)
    log_file = os.path.join(workspace_dir, "phase2_integrated_report.log")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SARA Engine Phase 2 Integrated Benchmark ===\n\n")

    print("=== [1] Hierarchical SDR Pooling (階層的特徴抽出) のテスト ===")
    pooling_layer = HierarchicalSDRPooling(in_size=1000, out_size=100, compression_ratio=0.5)
    
    # ダミーの低次スパイク入力 (例: 視覚のエッジ特徴)
    low_level_spikes = random.sample(range(1000), 150)
    
    start_time = time.time()
    # 学習モードで実行し、受容野を自己組織化させる
    high_level_spikes = pooling_layer.forward(low_level_spikes, learning=True)
    elapsed = (time.time() - start_time) * 1000
    
    msg_pooling = (
        f"Input Spikes Count: {len(low_level_spikes)}\n"
        f"Pooled Spikes Count (Abstracted): {len(high_level_spikes)}\n"
        f"Pooled Spikes: {high_level_spikes[:10]}...\n"
        f"Latency: {elapsed:.2f} ms\n"
    )
    print(msg_pooling)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("--- Pooling Test ---\n" + msg_pooling + "\n")


    print("=== [2] SDR Fuzzy Attention (曖昧さの許容) のテスト ===")
    attention_layer = SDRFuzzyAttention(sdr_size=1000, threshold=0.3)
    
    # メモリにKeyとValueを保存
    key_memory = [10, 20, 30, 40, 50, 60]
    value_memory = [100, 200, 300]
    attention_layer.forward(query=[], key=key_memory, value=value_memory)
    
    # 一部が欠損したクエリ（Fuzzy Query）で検索
    fuzzy_query = [10, 20, 30, 999] # Keyと部分一致
    
    start_time = time.time()
    recalled_value = attention_layer.forward(query=fuzzy_query)
    elapsed = (time.time() - start_time) * 1000
    
    msg_attn = (
        f"Fuzzy Query: {fuzzy_query}\n"
        f"Recalled Value (Should overlap with {value_memory}): {recalled_value}\n"
        f"Latency: {elapsed:.2f} ms\n"
    )
    print(msg_attn)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("--- Fuzzy Attention Test ---\n" + msg_attn + "\n")


    print("=== [3] Spiking Multimodal Hub (マルチソース統合) のテスト ===")
    hub = SpikingMultimodalHub(modalities=["vision", "text"], shared_space_size=2000)
    
    # 視覚とテキストの同期スパイク入力 (例: 「リンゴ」の画像と、「リンゴ」というテキスト)
    vision_spikes = [5, 15, 25, 105, 205]
    text_spikes = [10, 20, 30, 110, 210]
    
    print("学習フェーズ: 視覚とテキストを同期入力してヘッブ則で結合を強化...")
    for _ in range(3): # 複数回提示して結合を強める
        hub.forward({"vision": vision_spikes, "text": text_spikes}, learning=True)
    
    print("\n推論(想起)フェーズ: 視覚入力のみからテキストを想起...")
    start_time = time.time()
    # テキストを意図的に欠損させる
    _, predictions = hub.forward({"vision": vision_spikes}, learning=False)
    elapsed = (time.time() - start_time) * 1000
    
    recalled_text = predictions["text"]
    
    msg_hub = (
        f"Input Vision Spikes: {vision_spikes}\n"
        f"Recalled Text Spikes (Should match {text_spikes}): {recalled_text}\n"
        f"Latency: {elapsed:.2f} ms\n"
    )
    print(msg_hub)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("--- Multimodal Hub Test ---\n" + msg_hub + "\n")
        f.write(f"Log saved successfully to {log_file}\n")

    print(f"\nすべてのベンチマークが完了しました。ログは {log_file} に保存されました。")

if __name__ == "__main__":
    run_benchmarks()