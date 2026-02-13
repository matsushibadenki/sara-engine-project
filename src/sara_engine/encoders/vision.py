_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/vision.py",
    "//": "タイトル: 視覚エンコーダ (ImageSpikeEncoder)",
    "//": "目的: 画像データをスパイク列に変換する（レート/レイテンシコーディング）。"
}

import numpy as np
from typing import List, Tuple, Optional

class ImageSpikeEncoder:
    """
    画像をSNN用のスパイク列に変換するエンコーダ。
    網膜（Retina）の機能を模倣します。
    """
    def __init__(self, shape: Tuple[int, int] = (28, 28)):
        self.height, self.width = shape
        self.num_neurons = self.height * self.width

    def encode_rate(self, image: np.ndarray, time_steps: int = 20, max_rate: float = 0.8) -> List[List[int]]:
        """
        レートコーディング (Rate Coding)
        画素の明るさを「発火確率」に変換します。
        明るい場所ほど、時間内により多くのスパイクが発生します。
        """
        # 画像をフラット化し、0.0-1.0に正規化
        flat_img = image.flatten().astype(np.float32)
        if flat_img.max() > 1.0:
            flat_img /= 255.0
            
        spike_train = []
        
        for _ in range(time_steps):
            # 各ステップで確率的に発火判定
            # (画素値 * max_rate) > ランダム値 なら発火
            thresholds = flat_img * max_rate
            rand_vals = np.random.rand(self.num_neurons)
            fired_indices = np.where(rand_vals < thresholds)[0].tolist()
            spike_train.append(fired_indices)
            
        return spike_train

    def encode_latency(self, image: np.ndarray, time_steps: int = 20) -> List[List[int]]:
        """
        レイテンシコーディング (Latency/Time-to-First-Spike Coding)
        画素の明るさを「発火タイミング」に変換します。
        明るい場所ほど「早く」発火し、暗い場所は「遅く」発火します。
        """
        flat_img = image.flatten().astype(np.float32)
        if flat_img.max() > 1.0:
            flat_img /= 255.0
            
        spike_train = [[] for _ in range(time_steps)]
        
        # 0(黒)は発火しない、1(白)はstep 0で発火
        # 閾値を設け、それ以上明るいピクセルのみ処理
        active_pixels = np.where(flat_img > 0.1)[0]
        
        for idx in active_pixels:
            brightness = flat_img[idx]
            # 明るい(1.0) -> fire_time = 0
            # 暗い(0.1) -> fire_time = time_steps - 1
            fire_time = int((1.0 - brightness) * (time_steps - 1))
            fire_time = np.clip(fire_time, 0, time_steps - 1)
            
            # 指定された時間に発火を追加
            spike_train[fire_time].append(int(idx))
            
        return spike_train