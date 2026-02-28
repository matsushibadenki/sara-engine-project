_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/board.py",
    "//": "ファイルの日本語タイトル: Sara-Board (ダイナミクス可視化)",
    "//": "ファイルの目的や内容: TensorBoardの代替として、SNNのスパイク発火(ラスタープロット)やネットワークの動的状態を記録・可視化するツール。"
}

import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class SaraBoard:
    def __init__(self, log_dir: str = "workspace/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        # mypy対応: 変数の型を明示
        self.spike_records: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.current_step: Dict[str, int] = defaultdict(int)

    def log_spikes(self, layer_name: str, spikes: List[int]) -> None:
        """指定したレイヤーの発火ニューロンIDリストを現在のタイムステップとして記録する"""
        step = self.current_step[layer_name]
        for neuron_id in spikes:
            self.spike_records[layer_name].append((step, neuron_id))
        self.current_step[layer_name] += 1

    def save_records(self, filename: str = "spikes.json") -> None:
        """可視化用に生のスパイク履歴をJSONとして保存する"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.spike_records, f)

    def plot_raster(self, layer_name: str, save_name: str = "raster_plot.png") -> None:
        """記録されたスパイクデータから生体脳の観測で用いられるラスタープロットを生成する"""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib is not installed. Plotting is disabled. Please run `pip install matplotlib`.")
            return

        records = self.spike_records.get(layer_name, [])
        if not records:
            print(f"No spike data recorded for layer: {layer_name}")
            return

        steps = [r[0] for r in records]
        neurons = [r[1] for r in records]

        plt.figure(figsize=(10, 6))
        plt.scatter(steps, neurons, marker='|', color='black', s=50)
        plt.title(f"Sara-Board Raster Plot: {layer_name}")
        plt.xlabel("Time Step (Forward Pass)")
        plt.ylabel("Neuron ID")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        filepath = os.path.join(self.log_dir, save_name)
        plt.savefig(filepath)
        plt.close()
        print(f"Raster plot saved to {filepath}")