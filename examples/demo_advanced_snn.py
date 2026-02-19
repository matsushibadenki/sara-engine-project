import sys
import os
import time
import struct
import urllib.request
import gzip
import random

_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_advanced_snn.py",
    "//": "タイトル: SARA-Engine 高度なSNN学習デモ (Fashion-MNIST / 高解像度TTFS・90%限界突破版)",
    "//": "目的: TTFSのステップ数を5段階に高解像度化し、3層のマルチスケール受容野とスキップ結合を組み合わせることで、誤差逆伝播法なしで精度90%到達を目指す。"
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import ReadoutLayer

def download_data(filename, source_url, data_dir="data"):
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(filepath):
        print(f"[Data] ダウンロード中: {filename} ...")
        urllib.request.urlretrieve(source_url, filepath)
        print(f"[Data] ダウンロード完了: {filename}")
    return filepath

def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = []
        for i in range(num):
            start = i * rows * cols
            end = start + rows * cols
            images.append([int(b) for b in data[start:end]])
    return images

def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        return [int(b) for b in data]

def prepare_dataset(dataset_name="mnist"):
    if dataset_name == "mnist":
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    elif dataset_name == "fashion_mnist":
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    else:
        raise ValueError("Unknown dataset")

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

def apply_spatial_receptive_fields(layer, width, height, patch_sizes):
    if layer.use_rust and hasattr(layer.core, 'apply_spatial_receptive_fields'):
        layer.core.apply_spatial_receptive_fields(width, height, patch_sizes)
        return

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
                    w = 0.0
                    if pattern_type == 'random': w = random.uniform(-4.0, 4.0)
                    elif pattern_type == 'center_on': w = 6.0 if dx*dx + dy*dy <= (half_p*0.6)**2 else -3.0
                    elif pattern_type == 'center_off': w = -6.0 if dx*dx + dy*dy <= (half_p*0.6)**2 else 3.0
                    elif pattern_type == 'edge_h': w = 6.0 if dy < 0 else -6.0
                    elif pattern_type == 'edge_v': w = 6.0 if dx < 0 else -6.0
                    elif pattern_type == 'edge_d1': w = 6.0 if dx > dy else -6.0
                    elif pattern_type == 'edge_d2': w = 6.0 if dx > -dy else -6.0
                        
                    w += random.uniform(-0.8, 0.8) 
                    layer.in_weights[inp_idx][hidden_idx] = w * 1.5

def run_advanced_snn():
    print("="*65)
    print("SARA-Engine: Advanced SNN (High-Res TTFS / Limit Break)")
    print("="*65)

    DATASET_NAME = "fashion_mnist"
    SAVE_MODEL_PATH = f"sara_{DATASET_NAME}_model_batch_v3.json"

    train_images, train_labels, test_images, test_labels = prepare_dataset(DATASET_NAME)
    
    num_inputs = 784
    num_classes = 10
    
    print("\n[Network] 3層マルチスケール型 生物学的ネットワーク(Rust)を構築中...")
    
    # 微細・中間・大局の3層構造で多様な特徴を抽出
    layer_configs = [
        {"size": 3000, "decay": 0.80, "patch_sizes": [3, 5, 7]},
        {"size": 2500, "decay": 0.88, "patch_sizes": [9, 11, 13]},
        {"size": 2500, "decay": 0.95, "patch_sizes": [15, 17, 21]} 
    ]
    
    liquid_layers = []
    for cfg in layer_configs:
        layer = DynamicLiquidLayer(
            input_size=num_inputs, hidden_size=cfg["size"], 
            decay=cfg["decay"], target_rate=0.06, density=0.0, input_scale=0.0
        )
        apply_spatial_receptive_fields(layer, 28, 28, cfg["patch_sizes"])
        liquid_layers.append(layer)
    
    total_hidden = sum(cfg["size"] for cfg in layer_configs)
    
    # TTFSの解像度を5段階に引き上げ、中間階調の情報を保持
    thresholds = [212, 170, 128, 86, 44]
    
    temporal_spatial_size = (num_inputs + total_hidden) * len(thresholds)
    
    readout_layer = ReadoutLayer(
        input_size=temporal_spatial_size, 
        output_size=num_classes,
        learning_rate=0.003 # 次元が増大したため初期学習率を下げて安定させる
    )

    start_time = time.time()
    epochs = 6 
    print(f"\n[Train] 高解像度バッチ学習を開始します (データ数: {len(train_images)}件 x {epochs}エポック)...")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        combined = list(zip(train_images, train_labels))
        random.shuffle(combined)
        train_images_shuffled, train_labels_shuffled = zip(*combined)
        
        # 学習率の減衰を少し緩やかにし、終盤まで学習能力を維持
        if epoch > 0: 
            readout_layer.lr *= 0.8  
            
        epoch_start_time = time.time()
        for i in range(len(train_images)):
            image, target_label = train_images_shuffled[i], train_labels_shuffled[i]
            
            for l in liquid_layers:
                if l.use_rust and hasattr(l.core, 'reset_potentials'): 
                    l.core.reset_potentials()
                else: 
                    l.v, l.refractory = [0.0]*l.size, [0.0]*l.size
            
            accumulated_fired = set()
            prev_fired = [[] for _ in range(len(liquid_layers))]
            has_fired_inputs = set()
            
            for step, thresh in enumerate(thresholds):
                active_inputs = [idx for idx, p in enumerate(image) if p > thresh and idx not in has_fired_inputs]
                if not active_inputs: continue
                has_fired_inputs.update(active_inputs)
                
                time_offset = step * (num_inputs + total_hidden)
                accumulated_fired.update([idx + time_offset for idx in active_inputs])
                
                offset = num_inputs
                for l_idx, l in enumerate(liquid_layers):
                    fired_h = l.forward_with_feedback(active_inputs=active_inputs, prev_active_hidden=prev_fired[l_idx])
                    prev_fired[l_idx] = fired_h
                    accumulated_fired.update([f + offset + time_offset for f in fired_h])
                    offset += l.size
                
            if accumulated_fired:
                readout_layer.train_step(list(accumulated_fired), target_label)
                
            if (i + 1) % 15000 == 0: 
                print(f"  Processed {i + 1}/{len(train_images)} samples...")
        
        print(f"  -> エポック所要時間: {time.time() - epoch_start_time:.2f}秒")
        
        if epoch < epochs - 1:
            print("  " + readout_layer.sleep_phase(prune_rate=0.002))

    print(f"\n全学習完了 | 総所要時間: {time.time() - start_time:.2f}秒")
    print(readout_layer.sleep_phase(prune_rate=0.02))

    readout_layer.save_model(SAVE_MODEL_PATH)
    
    print("\n[Model] 保存したモデルを読み込んでテストします...")
    new_readout = ReadoutLayer(input_size=temporal_spatial_size, output_size=num_classes)
    new_readout.load_model(SAVE_MODEL_PATH)

    print(f"\n[Test] 最終推論テスト開始 (テストデータ数: {len(test_images)}件)...")
    correct_count = 0
    test_start_time = time.time()
    
    for i in range(len(test_images)):
        image = test_images[i]
        true_label = test_labels[i]
        
        for l in liquid_layers:
            if l.use_rust and hasattr(l.core, 'reset_potentials'): l.core.reset_potentials()
            else: l.v, l.refractory = [0.0]*l.size, [0.0]*l.size
            
        accumulated_fired = set()
        prev_fired = [[] for _ in range(len(liquid_layers))]
        has_fired_inputs = set()
        
        for step, thresh in enumerate(thresholds):
            active_inputs = [idx for idx, p in enumerate(image) if p > thresh and idx not in has_fired_inputs]
            if not active_inputs: continue
            has_fired_inputs.update(active_inputs)
            
            time_offset = step * (num_inputs + total_hidden)
            accumulated_fired.update([idx + time_offset for idx in active_inputs])
            
            offset = num_inputs
            for l_idx, l in enumerate(liquid_layers):
                fired_h = l.forward_with_feedback(active_inputs=active_inputs, prev_active_hidden=prev_fired[l_idx])
                prev_fired[l_idx] = fired_h
                accumulated_fired.update([f + offset + time_offset for f in fired_h])
                offset += l.size
            
        if accumulated_fired:
            potentials = new_readout.predict(list(accumulated_fired))
            predicted = potentials.index(max(potentials))
            if predicted == true_label:
                correct_count += 1
                
        if (i + 1) % 2000 == 0:
            print(f"  Tested {i + 1} | Current Accuracy: {(correct_count / (i + 1)) * 100:.1f}%")

    final_accuracy = (correct_count / len(test_images)) * 100
    print("="*65)
    print(f"最終テスト精度 ({DATASET_NAME}): {final_accuracy:.2f}%")
    print(f"推論所要時間: {time.time() - test_start_time:.2f}秒")
    print("="*65)

if __name__ == "__main__":
    run_advanced_snn()