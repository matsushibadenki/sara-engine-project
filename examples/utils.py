# examples/utils.py
# 共通ユーティリティ（パス設定、データ変換）

import sys
import os
import numpy as np

def setup_path():
    """
    srcディレクトリへのパスをsys.pathに追加し、モジュールのインポートを可能にする。
    戻り値: srcディレクトリの絶対パス
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return src_dir

def img_to_poisson(img_flat, time_steps=60, rate_scale=0.5):
    """
    画像をポアソンスパイク列に変換する（レートコーディング）
    """
    # 負の値をクリップし、正規化
    img_flat = np.maximum(0, img_flat)
    max_val = np.max(img_flat)
    if max_val > 1e-6:
        img_flat = img_flat / max_val
    
    rate = img_flat * rate_scale
    spike_train = []
    
    for _ in range(time_steps):
        # 乱数とピクセル強度を比較して発火判定
        fired = np.where(np.random.rand(len(img_flat)) < rate)[0].tolist()
        spike_train.append(fired)
        
    return spike_train

def text_to_spikes(text, vocab_map, steps_per_char=3, echo_steps=10):
    """
    テキストをスパイク列に変換する
    """
    spike_train = []
    
    for char in text:
        if char in vocab_map:
            neuron_idx = vocab_map[char]
            for _ in range(steps_per_char):
                spike_train.append([neuron_idx])
        else:
            spike_train.append([])
            
    # 文末の余韻（Echo）
    for _ in range(echo_steps):
        spike_train.append([])
        
    return spike_train

def load_mnist_data(data_dir='./data'):
    """
    MNISTデータをロードする（torchvisionが必要）
    """
    try:
        import torch
        # 修正: type: ignore 追加
        from torchvision import datasets, transforms # type: ignore
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        os.makedirs(data_dir, exist_ok=True)
        train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_dir, train=False, transform=transform)
        return train, test
    except ImportError:
        print("Error: 'torch' and 'torchvision' are required for MNIST.")
        print("Please install via: pip install torch torchvision")
        sys.exit(1)