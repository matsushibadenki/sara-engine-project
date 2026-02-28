# [配置するディレクトリのパス]: ./examples/demo_advanced_snn.py
# [ファイルの日本語タイトル]: SARA-Engine 高度なSNN学習デモ (Fashion-MNIST / Extreme Sparsity & Dropout)
# [ファイルの目的や内容]: 極限のスパース化（発火率1.5%）と強烈なSpike Dropout（35%）を組み合わせ、高次元空間での直交性を最大化。誤差逆伝播を一切用いずにFashion-MNISTで精度90%超えを達成する。
_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_advanced_snn.py",
    "//": "タイトル: SARA-Engine 高度なSNN学習デモ (Fashion-MNIST / Extreme Sparsity & Dropout)",
    "//": "目的: 極限のスパース化（発火率1.5%）と強烈なSpike Dropout（35%）を組み合わせ、高次元空間での直交性を最大化。誤差逆伝播を一切用いずにFashion-MNISTで精度90%超えを達成する。"
}

import sys
import os
import time
import struct
import urllib.request
import gzip
import random
import math
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import SpikeReadoutLayer

def download_data(filename: str, source_url: str, data_dir: str = "data") -> str:
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(filepath):
        print(f"[Data] ダウンロード中: {filename} ...")
        urllib.request.urlretrieve(source_url, filepath)
    return filepath

def load_images(filepath: str) -> list[list[int]]:
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images: list[list[int]] = []
        for i in range(num):
            start = i * rows * cols
            end = start + rows * cols
            images.append([int(b) for b in data[start:end]])
    return images

def load_labels(filepath: str) -> list[int]:
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        return [int(b) for b in data]

def prepare_dataset(dataset_name: str = "fashion_mnist") -> tuple[list[list[int]], list[int], list[list[int]], list[int]]:
    if dataset_name == "fashion_mnist":
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    else:
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    files = {
        "train_img": (f"{dataset_name}_train-images.gz", base_url + "train-images-idx3-ubyte.gz"),
        "train_lbl": (f"{dataset_name}_train-labels.gz", base_url + "train-labels-idx1-ubyte.gz"),
        "test_img": (f"{dataset_name}_t10k-images.gz", base_url + "t10k-images-idx3-ubyte.gz"),
        "test_lbl": (f"{dataset_name}_t10k-labels.gz", base_url + "t10k-labels-idx1-ubyte.gz"),
    }
    
    paths = {k: download_data(v[0], v[1]) for k, v in files.items()}
    
    print(f"\n[Data] {dataset_name} データセットを展開中...")
    train_images = load_images(paths["train_img"])
    train_labels = load_labels(paths["train_lbl"])
    test_images = load_images(paths["test_img"])
    test_labels = load_labels(paths["test_lbl"])
    
    return train_images, train_labels, test_images, test_labels

def apply_spatial_receptive_fields(layer: DynamicLiquidLayer, width: int, height: int, patch_sizes: list[int]) -> None:
    if layer.use_rust and hasattr(layer.core, 'apply_spatial_receptive_fields'):
        print(f"  -> 局所受容野（Spatial Receptive Fields: Gabor）をRustコア内に構築中... (層サイズ: {layer.size}, パッチ: {patch_sizes})")
        layer.core.apply_spatial_receptive_fields(width, height, patch_sizes) # type: ignore
        return

    print(f"  -> 局所受容野（Spatial Receptive Fields: Gabor）をPythonで構築中... (層サイズ: {layer.size})")
    layer.in_weights = [{} for _ in range(width * height)]
    
    angles = [0.0, math.pi / 4.0, math.pi / 2.0, math.pi * 3.0 / 4.0]
    phases = [0.0, math.pi / 2.0]
    
    for hidden_idx in range(layer.size):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        patch_size = random.choice(patch_sizes)
        half_p = patch_size // 2
        
        theta = random.choice(angles)
        psi = random.choice(phases)
        lambda_val = patch_size * 0.8
        sigma = patch_size * 0.4
        gamma = 0.5
        
        for dy in range(-half_p, half_p + 1):
            for dx in range(-half_p, half_p + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height:
                    inp_idx = y * width + x
                    
                    x_prime = dx * math.cos(theta) + dy * math.sin(theta)
                    y_prime = -dx * math.sin(theta) + dy * math.cos(theta)
                    
                    env = math.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2.0 * sigma**2))
                    carrier = math.cos(2.0 * math.pi * x_prime / lambda_val + psi)
                    
                    w = env * carrier * 10.0
                    w += random.uniform(-0.5, 0.5) 
                    
                    layer.in_weights[inp_idx][hidden_idx] = w

def preprocess_images(images: list[list[int]], thresholds: list[int]) -> list[list[list[int]]]:
    print(f"  -> Rank-Order エンコーディングの事前計算中 (閾値: {thresholds})...")
    preprocessed: list[list[list[int]]] = []
    min_thresh = thresholds[-1]
    num_steps = len(thresholds)
    
    for img in images:
        active_per_step: list[list[int]] = [[] for _ in range(num_steps)]
        for idx, pixel in enumerate(img):
            if pixel > min_thresh:
                for step in range(num_steps):
                    if pixel > thresholds[step]:
                        active_per_step[step].append(idx)
                        break
        preprocessed.append(active_per_step)
        
    return preprocessed

def save_snn_model(readout_layer: SpikeReadoutLayer, filepath: str) -> None:
    print(f"\n[Model] 学習済みシナプス重みを保存中: {filepath}")
    data = {
        "W": readout_layer.W,
        "b": readout_layer.b
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)
    print("  -> 保存完了")

def load_snn_model(readout_layer: SpikeReadoutLayer, filepath: str) -> None:
    print(f"\n[Model] シナプス重みをファイルから復元中: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    readout_layer.W = [{int(k): float(v) for k, v in layer_dict.items()} for layer_dict in data["W"]]
    readout_layer.b = data["b"]
    print("  -> 復元完了")

def run_advanced_snn() -> None:
    print("="*65)
    print("SARA-Engine: Advanced SNN (Fashion-MNIST / Extreme Sparsity & Dropout)")
    print("="*65)

    DATASET_NAME = "fashion_mnist"
    SAVE_MODEL_PATH = f"sara_{DATASET_NAME}_model_stdp.json"

    train_images, train_labels, test_images, test_labels = prepare_dataset(DATASET_NAME)
    
    num_inputs = 784
    num_classes = 10
    epochs = 3  # 安定した学習率でじっくりと最適解を探索する
    
    print("\n[Network] 生物学的マルチスケール・ネットワークを構築中...")
    
    # スケールごとに独立した3つの特徴抽出層を構築
    layer_configs = [
        {"size": 6000, "decay": 0.50, "patch_sizes": [3, 4]},      # 極小エッジ・微細なテクスチャ
        {"size": 6000, "decay": 0.70, "patch_sizes": [5, 6, 7]},   # 中間パーツ（襟、靴紐など）
        {"size": 6000, "decay": 0.90, "patch_sizes": [9, 12, 16]}  # マクロシルエット
    ]
    
    liquid_layers: list[DynamicLiquidLayer] = []
    for cfg in layer_configs:
        size_val = int(cfg["size"]) # type: ignore
        decay_val = float(cfg["decay"]) # type: ignore
        patch_sizes_val = cfg["patch_sizes"] # type: ignore
        layer = DynamicLiquidLayer(
            input_size=num_inputs, 
            hidden_size=size_val, 
            decay=decay_val, 
            target_rate=0.015, # 【重要】極限まで発火を絞り込み、表現の直交性を最大化
            density=0.0,
            input_scale=0.0
        )
        apply_spatial_receptive_fields(layer, 28, 28, patch_sizes_val) # type: ignore
        liquid_layers.append(layer)
    
    # 閾値をシンプルにして情報過多を防ぐ
    thresholds = [200, 130, 60, 20]
    num_steps = len(thresholds)
    
    total_hidden = sum(int(cfg["size"]) for cfg in layer_configs) # type: ignore
    feature_dim_per_step = num_inputs + total_hidden
    temporal_spatial_size = feature_dim_per_step * num_steps
    
    readout_layer = SpikeReadoutLayer(
        d_model=temporal_spatial_size, 
        vocab_size=num_classes,
        learning_rate=0.015,  # 【重要】マイルドな学習率で過学習（忘却）を防ぐ
        use_refractory=False
    )

    print("\n[Pre-process] 画像データの前処理...")
    train_encoded = preprocess_images(train_images, thresholds)
    test_encoded = preprocess_images(test_images, thresholds)

    train_samples_to_use = len(train_images)
    test_samples_to_use = len(test_images)

    print(f"\n[Train] 学習開始 (データ数: {train_samples_to_use}件 x {epochs}エポック)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        combined = list(zip(train_encoded, train_labels))
        random.shuffle(combined)
        
        # zip(*) 展開時の曖昧なジェネレータ型推論を回避するため、明示的リスト内包を使用
        train_encoded_shuffled: list[list[list[int]]] = [item[0] for item in combined]
        train_labels_shuffled: list[int] = [item[1] for item in combined]
        
        # エポックごとに学習率を緩やかに減衰
        if epoch > 0:
            readout_layer.learning_rate *= 0.85
            
        for i in range(train_samples_to_use):
            encoded_image = train_encoded_shuffled[i]
            target_label = train_labels_shuffled[i]
            
            for l in liquid_layers:
                if l.use_rust and hasattr(l.core, 'reset_potentials'):
                    l.core.reset_potentials() # type: ignore
                else:
                    l.v = [0.0] * l.size
                    l.refractory = [0.0] * l.size
                    new_thresh: list[float] = []
                    for t in l.dynamic_thresh:
                        new_thresh.append(5.0 if t > 5.0 else t)
                    l.dynamic_thresh = new_thresh
            
            accumulated_fired: set[int] = set()
            prev_fired: list[list[int]] = [[] for _ in range(len(liquid_layers))]
            hidden_already_fired: list[set[int]] = [set() for _ in range(len(liquid_layers))]
            
            for step, active_inputs in enumerate(encoded_image):
                if not active_inputs:
                    continue
                
                step_base_offset = step * feature_dim_per_step
                for inp in active_inputs:
                    accumulated_fired.add(inp + step_base_offset)
                
                hidden_offset = num_inputs
                for l_idx, l in enumerate(liquid_layers):
                    fired_hidden = l.forward(
                        active_inputs=active_inputs,
                        prev_active_hidden=prev_fired[l_idx]
                    )
                    prev_fired[l_idx] = fired_hidden
                    
                    first_fired: list[int] = []
                    for f in fired_hidden:
                        if f not in hidden_already_fired[l_idx]:
                            first_fired.append(f)
                            
                    if first_fired:
                        hidden_already_fired[l_idx].update(first_fired)
                        for f in first_fired:
                            accumulated_fired.add(f + step_base_offset + hidden_offset)
                        
                    hidden_offset += l.size
                
            if accumulated_fired:
                # 【重要】強烈な Spike Dropout (35%欠落)
                # これにより単一ピクセルや局所的な特徴への過度な依存を断ち切る
                train_spikes: list[int] = []
                for s in accumulated_fired:
                    if random.random() > 0.35:
                        train_spikes.append(s)
                readout_layer.forward(train_spikes, target_token=target_label, learning=True)
                
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{train_samples_to_use} samples...")

    print(f"学習完了 | 所要時間: {time.time() - start_time:.2f}秒")

    print("\n[Sleep] 睡眠フェーズ開始 (ノイズシナプスの枝刈り)...")
    pruned_count = 0
    total_count = 0
    prune_rate = 0.005 
    for s_idx in range(len(readout_layer.W)):
        to_delete: list[int] = []
        for t_id, weight in readout_layer.W[s_idx].items():
            if abs(weight) < prune_rate:
                to_delete.append(t_id)
                
        for t_id in to_delete:
            del readout_layer.W[s_idx][t_id]
            pruned_count += 1
            
        total_count += len(readout_layer.W[s_idx]) + len(to_delete)
        
    print(f"睡眠フェーズ完了: 全 {total_count} シナプス中 {pruned_count} 個を枝刈りしました。")

    # ---------------------------------------------------------
    # モデルの保存と読み込みのテスト
    # ---------------------------------------------------------
    save_snn_model(readout_layer, SAVE_MODEL_PATH)
    
    new_readout = SpikeReadoutLayer(
        d_model=temporal_spatial_size, 
        vocab_size=num_classes,
        learning_rate=0.0,
        use_refractory=False
    )
    load_snn_model(new_readout, SAVE_MODEL_PATH)

    print(f"\n[Test] 復元したモデルによる推論テスト開始 (テストデータ数: {test_samples_to_use}件)...")
    correct_count = 0
    test_start_time = time.time()
    
    for i in range(test_samples_to_use):
        encoded_image = test_encoded[i]
        true_label = test_labels[i]
        
        for l in liquid_layers:
            if l.use_rust and hasattr(l.core, 'reset_potentials'):
                l.core.reset_potentials() # type: ignore
            else:
                l.v = [0.0] * l.size
                l.refractory = [0.0] * l.size
            
        accumulated_fired = set()
        prev_fired = [[] for _ in range(len(liquid_layers))]
        hidden_already_fired = [set() for _ in range(len(liquid_layers))]
        
        for step, active_inputs in enumerate(encoded_image):
            if not active_inputs:
                continue
            
            step_base_offset = step * feature_dim_per_step
            for inp in active_inputs:
                accumulated_fired.add(inp + step_base_offset)
            
            hidden_offset = num_inputs
            for l_idx, l in enumerate(liquid_layers):
                fired_h = l.forward(
                    active_inputs=active_inputs,
                    prev_active_hidden=prev_fired[l_idx]
                )
                prev_fired[l_idx] = fired_h
                
                first_fired = []
                for f in fired_h:
                    if f not in hidden_already_fired[l_idx]:
                        first_fired.append(f)
                        
                if first_fired:
                    hidden_already_fired[l_idx].update(first_fired)
                    for f in first_fired:
                        accumulated_fired.add(f + step_base_offset + hidden_offset)
                    
                hidden_offset += l.size
            
        if accumulated_fired:
            # 推論時は Dropout なしで全スパイクを利用し、アンサンブル効果を得る
            predicted = new_readout.forward(list(accumulated_fired), learning=False)
            if predicted == true_label:
                correct_count += 1
                
        if (i + 1) % 2000 == 0:
            current_acc = (correct_count / (i + 1)) * 100
            print(f"  Tested {i + 1}/{test_samples_to_use} | Current Accuracy: {current_acc:.1f}%")

    final_accuracy = (correct_count / test_samples_to_use) * 100
    print("="*65)
    print(f"最終テスト精度 ({DATASET_NAME}): {final_accuracy:.2f}%")
    print(f"推論所要時間: {time.time() - test_start_time:.2f}秒")
    print("="*65)

if __name__ == "__main__":
    run_advanced_snn()