_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_mnist_snn.py",
    "//": "タイトル: SARA-Engine MNIST手書き数字認識デモ (時空間コーディング・95%突破版)",
    "//": "目的: TTFSの各時間ステップを空間次元に展開(Temporal-to-Spatial)し、受容野を強化することで、逆伝播なしで精度95%を超える。"
}

import sys
import os
import time
import struct
import urllib.request
import gzip
import random

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import ReadoutLayer

def download_mnist(filename, source_url):
    filepath = os.path.join("data", filename)
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(filepath):
        print(f"[Data] ダウンロード中: {filename} ...")
        urllib.request.urlretrieve(source_url, filepath)
        print(f"[Data] ダウンロード完了: {filename}")
    return filepath

def load_mnist_images(filepath):
    print(f"[Data] 画像を展開中: {filepath} ...")
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = []
        for i in range(num):
            start = i * rows * cols
            end = start + rows * cols
            images.append([int(b) for b in data[start:end]])
    return images

def load_mnist_labels(filepath):
    print(f"[Data] ラベルを展開中: {filepath} ...")
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        return [int(b) for b in data]

def prepare_mnist_dataset():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": ("train-images-idx3-ubyte.gz", base_url + "train-images-idx3-ubyte.gz"),
        "train_lbl": ("train-labels-idx1-ubyte.gz", base_url + "train-labels-idx1-ubyte.gz"),
        "test_img": ("t10k-images-idx3-ubyte.gz", base_url + "t10k-images-idx3-ubyte.gz"),
        "test_lbl": ("t10k-labels-idx1-ubyte.gz", base_url + "t10k-labels-idx1-ubyte.gz"),
    }
    
    paths = {k: download_mnist(v[0], v[1]) for k, v in files.items()}
    
    train_images = load_mnist_images(paths["train_img"])
    train_labels = load_mnist_labels(paths["train_lbl"])
    test_images = load_mnist_images(paths["test_img"])
    test_labels = load_mnist_labels(paths["test_lbl"])
    
    return train_images, train_labels, test_images, test_labels

def apply_spatial_receptive_fields(layer, width, height, patch_sizes):
    """
    視覚野（V1）の構造を模倣。斜め方向（Diagonal）のエッジ抽出も追加し表現力を強化。
    """
    print(f"  -> 局所受容野（Spatial Receptive Fields）を構築中... (層サイズ: {layer.size})")
    layer.in_weights = [{} for _ in range(width * height)]
    
    patterns = ['random', 'center_on', 'center_off', 'edge_h', 'edge_v', 'edge_d1', 'edge_d2']
    
    for hidden_idx in range(layer.size):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        patch_size = random.choice(patch_sizes)
        half_p = patch_size // 2
        
        pattern_type = random.choice(patterns)
        
        for dy in range(-half_p, half_p + 1):
            for dx in range(-half_p, half_p + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height:
                    inp_idx = y * width + x
                    
                    if pattern_type == 'random':
                        w = random.uniform(-4.0, 4.0)
                    elif pattern_type == 'center_on':
                        dist = dx*dx + dy*dy
                        w = 6.0 if dist <= (half_p*0.6)**2 else -3.0
                    elif pattern_type == 'center_off':
                        dist = dx*dx + dy*dy
                        w = -6.0 if dist <= (half_p*0.6)**2 else 3.0
                    elif pattern_type == 'edge_h':
                        w = 6.0 if dy < 0 else -6.0
                    elif pattern_type == 'edge_v':
                        w = 6.0 if dx < 0 else -6.0
                    elif pattern_type == 'edge_d1':
                        w = 6.0 if dx > dy else -6.0
                    elif pattern_type == 'edge_d2':
                        w = 6.0 if dx > -dy else -6.0
                        
                    w += random.uniform(-0.8, 0.8) # ノイズを少し増やして汎化
                    layer.in_weights[inp_idx][hidden_idx] = w * 1.5

def run_mnist_snn():
    print("="*60)
    print("SARA-Engine: MNIST Classification (Temporal-Spatial & SRF)")
    print("="*60)

    train_images, train_labels, test_images, test_labels = prepare_mnist_dataset()
    
    num_inputs = 784
    num_classes = 10
    epochs = 3 
    
    train_samples_to_use = len(train_images)
    test_samples_to_use = len(test_images)

    print("\n[Network] 生物学的マルチスケール・ネットワークを構築中...")
    
    # 特徴抽出能力を高めるため、ニューロン数を少し増加 (合計5000)
    layer_configs = [
        {"size": 3000, "decay": 0.80, "patch_sizes": [3, 5, 7]},
        {"size": 2000, "decay": 0.95, "patch_sizes": [9, 13, 17]} 
    ]
    
    liquid_layers = []
    for cfg in layer_configs:
        layer = DynamicLiquidLayer(
            input_size=num_inputs, 
            hidden_size=cfg["size"], 
            decay=cfg["decay"], 
            target_rate=0.06,
            density=0.0,
            input_scale=0.0,
            use_rust=False
        )
        apply_spatial_receptive_fields(layer, 28, 28, cfg["patch_sizes"])
        liquid_layers.append(layer)
    
    total_hidden = sum(cfg["size"] for cfg in layer_configs)
    
    # 時間ステップ（3段階へ戻し、時間情報を確保）
    thresholds = [200, 128, 64]
    
    # Temporal-to-Spatial Encoding:
    # 時間ステップごとの発火を別々の入力次元（Phase）として扱う。
    # これにより線形分離器が「早く発火したのか、遅く発火したのか」を区別できるようになる。
    temporal_spatial_size = total_hidden * len(thresholds)
    
    readout_layer = ReadoutLayer(
        input_size=temporal_spatial_size, 
        output_size=num_classes,
        learning_rate=0.005
    )

    print(f"\n[Train] 学習開始 (データ数: {train_samples_to_use}件 x {epochs}エポック)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        combined = list(zip(train_images, train_labels))
        random.shuffle(combined)
        train_images_shuffled, train_labels_shuffled = zip(*combined)
        
        readout_layer.t = 0
        for o in range(readout_layer.output_size):
            for idx in readout_layer.m[o]:
                readout_layer.m[o][idx] = 0.0
                readout_layer.v[o][idx] = 0.0
        
        if epoch > 0:
            readout_layer.lr *= 0.6  
            
        for i in range(train_samples_to_use):
            image = train_images_shuffled[i]
            target_label = train_labels_shuffled[i]
            
            for l in liquid_layers:
                l.v = [0.0] * l.size
                l.refractory = [0.0] * l.size
                l.dynamic_thresh = [5.0 if t > 5.0 else t for t in l.dynamic_thresh] # 上限を5.0へ緩和
            
            accumulated_fired = set()
            prev_fired = [[] for _ in range(len(liquid_layers))]
            has_fired_inputs = set()
            
            for step, thresh in enumerate(thresholds):
                active_inputs = [
                    idx for idx, pixel in enumerate(image) 
                    if pixel > thresh and idx not in has_fired_inputs
                ]
                if not active_inputs:
                    continue
                    
                has_fired_inputs.update(active_inputs)
                
                offset = 0
                time_offset = step * total_hidden # タイムステップによる空間IDのシフト
                
                for l_idx, l in enumerate(liquid_layers):
                    fired_hidden = l.forward_with_feedback(
                        active_inputs=active_inputs,
                        prev_active_hidden=prev_fired[l_idx]
                    )
                    prev_fired[l_idx] = fired_hidden
                    
                    # 時間情報（Phase）を空間IDに変換して保存
                    accumulated_fired.update([f + offset + time_offset for f in fired_hidden])
                    offset += l.size
                
            if accumulated_fired:
                readout_layer.train_step(list(accumulated_fired), target_label)
                
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{train_samples_to_use} samples...")

    print(f"学習完了 | 所要時間: {time.time() - start_time:.2f}秒")

    print("\n[Sleep] 睡眠フェーズ開始 (ノイズシナプスの枝刈り)...")
    print(readout_layer.sleep_phase(prune_rate=0.03))

    print(f"\n[Test] 推論テスト開始 (テストデータ数: {test_samples_to_use}件)...")
    correct_count = 0
    test_start_time = time.time()
    
    for i in range(test_samples_to_use):
        image = test_images[i]
        true_label = test_labels[i]
        
        for l in liquid_layers:
            l.v = [0.0] * l.size
            l.refractory = [0.0] * l.size
            
        accumulated_fired = set()
        prev_fired = [[] for _ in range(len(liquid_layers))]
        has_fired_inputs = set()
        
        for step, thresh in enumerate(thresholds):
            active_inputs = [
                idx for idx, pixel in enumerate(image) 
                if pixel > thresh and idx not in has_fired_inputs
            ]
            if not active_inputs:
                continue
                
            has_fired_inputs.update(active_inputs)
            
            offset = 0
            time_offset = step * total_hidden
            
            for l_idx, l in enumerate(liquid_layers):
                fired_h = l.forward_with_feedback(
                    active_inputs=active_inputs,
                    prev_active_hidden=prev_fired[l_idx]
                )
                prev_fired[l_idx] = fired_h
                accumulated_fired.update([f + offset + time_offset for f in fired_h])
                offset += l.size
            
        if accumulated_fired:
            potentials = readout_layer.predict(list(accumulated_fired))
            predicted = potentials.index(max(potentials))
            if predicted == true_label:
                correct_count += 1
                
        if (i + 1) % 2000 == 0:
            current_acc = (correct_count / (i + 1)) * 100
            print(f"  Tested {i + 1}/{test_samples_to_use} | Current Accuracy: {current_acc:.1f}%")

    final_accuracy = (correct_count / test_samples_to_use) * 100
    print("="*60)
    print(f"最終テスト精度: {final_accuracy:.2f}%")
    print(f"推論所要時間: {time.time() - test_start_time:.2f}秒")
    print("="*60)

if __name__ == "__main__":
    run_mnist_snn()