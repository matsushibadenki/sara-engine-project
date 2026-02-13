_FILE_INFO = {
    "//": "ディレクトリパス: examples/utils.py",
    "//": "タイトル: 共通ユーティリティ（データ変換・ロード）",
    "//": "目的: インストール済みのライブラリと組み合わせて使用するデータ処理用ユーティリティ。"
}

import os
import sys
import numpy as np

def img_to_poisson(img_flat, time_steps=60, rate_scale=0.5):
    """
    画像をポアソンスパイク列に変換する（レートコーディング）
    """
    img_flat = np.maximum(0, img_flat)
    max_val = np.max(img_flat)
    if max_val > 1e-6:
        img_flat = img_flat / max_val
    
    rate = img_flat * rate_scale
    spike_train = []
    
    for _ in range(time_steps):
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
            
    for _ in range(echo_steps):
        spike_train.append([])
        
    return spike_train

def load_mnist_data(data_dir='./data'):
    """
    MNISTデータをロードする（torchvisionが必要）
    """
    try:
        import torch
        from torchvision import datasets, transforms
        
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