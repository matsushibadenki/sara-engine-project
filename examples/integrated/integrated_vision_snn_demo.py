# [配置するディレクトリのパス]: examples/integrated_vision_snn_demo.py
# [ファイルの日本語タイトル]: 統合SNN画像解析デモ (MNIST/Fashion-MNIST対応)
# [ファイルの目的や内容]: 複数の画像分類デモを統合。V1単純細胞を模したGaborフィルタ、Rank-Order符号化、強烈なSpike Dropout、および睡眠（枝刈り）フェーズを組み合わせ、行列演算なしで高精度な認識を実現する。

import sys
import os
import time
import struct
import urllib.request
import gzip
import random
import math
import json

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import SpikeReadoutLayer

def download_data(filename: str, source_url: str, data_dir: str = "data") -> str:
    """データのダウンロード"""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(filepath):
        print(f"[Data] ダウンロード中: {filename} ...")
        urllib.request.urlretrieve(source_url, filepath)
    return filepath

def load_images(filepath: str) -> list[list[int]]:
    """GZIP形式の画像データ展開"""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        return [[int(b) for b in data[i*rows*cols : (i+1)*rows*cols]] for i in range(num)]

def load_labels(filepath: str) -> list[int]:
    """GZIP形式のラベルデータ展開"""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return [int(b) for b in f.read()]

def prepare_dataset(dataset_type: str = "mnist"):
    """データセットの準備"""
    if dataset_type == "fashion":
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        prefix = "fashion_mnist"
    else:
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        prefix = ""

    files = {
        "train_img": (f"{prefix}train-images-idx3-ubyte.gz", base_url + "train-images-idx3-ubyte.gz"),
        "train_lbl": (f"{prefix}train-labels-idx1-ubyte.gz", base_url + "train-labels-idx1-ubyte.gz"),
        "test_img": (f"{prefix}t10k-images-idx3-ubyte.gz", base_url + "t10k-images-idx3-ubyte.gz"),
        "test_lbl": (f"{prefix}t10k-labels-idx1-ubyte.gz", base_url + "t10k-labels-idx1-ubyte.gz"),
    }
    
    paths = {k: download_data(v[0], v[1]) for k, v in files.items()}
    return load_images(paths["train_img"]), load_labels(paths["train_lbl"]), \
           load_images(paths["test_img"]), load_labels(paths["test_lbl"])

def apply_gabor_receptive_fields(layer: DynamicLiquidLayer, width: int, height: int, patch_sizes: list[int]):
    """V1受容野（Gaborフィルタ）の構築"""
    print(f"  -> 局所受容野（Gaborフィルタ）を構築中... (層サイズ: {layer.size})")
    layer.in_weights = [{} for _ in range(width * height)]
    angles = [0.0, math.pi/4, math.pi/2, math.pi*3/4]
    
    for hidden_idx in range(layer.size):
        cx, cy = random.randint(0, width-1), random.randint(0, height-1)
        patch_size = random.choice(patch_sizes)
        half_p = patch_size // 2
        theta, psi = random.choice(angles), random.choice([0.0, math.pi/2])
        
        for dy in range(-half_p, half_p + 1):
            for dx in range(-half_p, half_p + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < width and 0 <= y < height:
                    x_p = dx * math.cos(theta) + dy * math.sin(theta)
                    y_p = -dx * math.sin(theta) + dy * math.cos(theta)
                    env = math.exp(-(x_p**2 + 0.5**2 * y_p**2) / (2.0 * (patch_size*0.4)**2))
                    carrier = math.cos(2.0 * math.pi * x_p / (patch_size*0.8) + psi)
                    layer.in_weights[y * width + x][hidden_idx] = env * carrier * 10.0

def preprocess_rank_order(images: list[list[int]], thresholds: list[int]):
    """Rank-Order符号化による時間軸スパイク生成"""
    print(f"  -> Rank-Order符号化（閾値: {thresholds}）を計算中...")
    preprocessed = []
    min_t = thresholds[-1]
    for img in images:
        steps = [[] for _ in range(len(thresholds))]
        for idx, pixel in enumerate(img):
            if pixel > min_t:
                for s, t in enumerate(thresholds):
                    if pixel > t:
                        steps[s].append(idx)
                        break
        preprocessed.append(steps)
    return preprocessed

def run_integrated_demo(dataset_type="mnist", epochs=3):
    print("="*65)
    print(f"SARA-Engine: Integrated Vision SNN Demo ({dataset_type.upper()})")
    print("="*65)

    train_img, train_lbl, test_img, test_lbl = prepare_dataset(dataset_type)
    
    # ネットワーク設定
    layer_configs = [
        {"size": 5000, "decay": 0.50, "patches": [4, 6]},
        {"size": 5000, "decay": 0.80, "patches": [10, 15]}
    ]
    
    liquid_layers = []
    for cfg in layer_configs:
        layer = DynamicLiquidLayer(input_size=784, hidden_size=cfg["size"], decay=cfg["decay"], target_rate=0.015)
        apply_gabor_receptive_fields(layer, 28, 28, cfg["patches"])
        liquid_layers.append(layer)
    
    thresholds = [200, 130, 60, 20]
    total_hidden = sum(c["size"] for c in layer_configs)
    feat_dim = (784 + total_hidden) * len(thresholds)
    
    readout = SpikeReadoutLayer(d_model=feat_dim, vocab_size=10, learning_rate=0.015)

    train_enc = preprocess_rank_order(train_img[:20000], thresholds) # 時短のため制限
    test_enc = preprocess_rank_order(test_img[:2000], thresholds)

    print(f"\n[Train] 学習開始...")
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        for i, (encoded_img, target) in enumerate(zip(train_enc, train_lbl)):
            for l in liquid_layers: l.reset()
            accumulated = set()
            
            for step, active_in in enumerate(encoded_img):
                if not active_in: continue
                offset = step * (784 + total_hidden)
                for inp in active_in: accumulated.add(inp + offset)
                
                h_offset = 784
                for l_idx, l in enumerate(liquid_layers):
                    fired = l.forward(active_inputs=active_in, prev_active_hidden=[])
                    for f in fired: accumulated.add(f + offset + h_offset)
                    h_offset += l.size
            
            if accumulated:
                # Spike Dropout (35%欠落) による堅牢化
                train_spikes = [s for s in accumulated if random.random() > 0.35]
                readout.forward(train_spikes, target_token=target, learning=True)
            
            if (i+1) % 5000 == 0: print(f"  Processed {i+1} samples...")

    # 睡眠フェーズ（枝刈り）
    print("\n[Sleep] 睡眠フェーズ（低荷重シナプスの枝刈り）...")
    pruned = 0
    for s_idx in range(len(readout.W)):
        to_del = [t for t, w in readout.W[s_idx].items() if abs(w) < 0.005]
        for t in to_del:
            del readout.W[s_idx][t]
            pruned += 1
    print(f"  -> {pruned} 個のシナプスを枝刈りしました。")

    print(f"\n[Test] 推論テスト開始...")
    correct = 0
    for i, (encoded_img, true_lbl) in enumerate(zip(test_enc, test_lbl)):
        for l in liquid_layers: l.reset()
        accumulated = set()
        for step, active_in in enumerate(encoded_img):
            offset = step * (784 + total_hidden)
            for inp in active_in: accumulated.add(inp + offset)
            h_offset = 784
            for l in liquid_layers:
                fired = l.forward(active_inputs=active_in, prev_active_hidden=[])
                for f in fired: accumulated.add(f + offset + h_offset)
                h_offset += l.size
        
        if accumulated:
            if readout.forward(list(accumulated), learning=False) == true_lbl:
                correct += 1
    
    print("="*65)
    print(f"最終テスト精度: {(correct/len(test_enc))*100:.2f}%")
    print("="*65)

if __name__ == "__main__":
    run_integrated_demo(dataset_type="mnist") # または "fashion"