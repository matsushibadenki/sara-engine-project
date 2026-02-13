_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/visualizer.py",
    "//": "タイトル: SARA可視化ツールキット",
    "//": "目的: スパイク、膜電位、アテンションの可視化機能を提供するユーティリティ。"
}

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import os

class SaraVisualizer:
    """
    SNNの内部状態を可視化するためのツールキット
    ROADMAP Item 1: ニューロモーフィック・デバッグ & 可視化ツール
    """
    def __init__(self, save_dir: str = "logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_raster(self, spike_history: List[List[int]], title: str = "Spike Raster Plot", filename: str = "raster.png"):
        """
        スパイク・ラスタープロットの生成
        時間軸に沿ったニューロンの発火をプロットします。
        """
        plt.figure(figsize=(12, 6))
        
        times = []
        neurons = []
        
        for t, spikes in enumerate(spike_history):
            for neuron_idx in spikes:
                times.append(t)
                neurons.append(neuron_idx)
                
        if times:
            plt.scatter(times, neurons, s=1.5, c='black', alpha=0.6)
        
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Neuron Index")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        path = f"{self.save_dir}/{filename}"
        plt.savefig(path)
        plt.close()
        print(f"[Viz] Saved raster plot to {path}")

    def plot_membrane_potential_distribution(self, potentials: np.ndarray, thresholds: np.ndarray, filename: str = "potential_dist.png"):
        """
        膜電位分布の統計分析
        ニューロンの膜電位と閾値の分布をヒストグラムで表示します。
        """
        plt.figure(figsize=(10, 5))
        
        plt.hist(potentials, bins=50, alpha=0.5, label='Membrane Potential', color='blue')
        plt.hist(thresholds, bins=50, alpha=0.5, label='Dynamic Threshold', color='red')
        
        plt.title("Membrane Potential vs Threshold Distribution")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        
        path = f"{self.save_dir}/{filename}"
        plt.savefig(path)
        plt.close()
        print(f"[Viz] Saved potential distribution to {path}")

    def plot_attention_heatmap(self, attention_history: List[List[int]], memory_size: int, filename: str = "attention_map.png"):
        """
        アテンション・ヒートマップ
        どのメモリ（過去の文脈）にアクセスしたかをヒートマップで表示します。
        """
        if not attention_history:
            print("[Viz] Warning: No attention history to plot.")
            return

        steps = len(attention_history)
        heatmap = np.zeros((steps, memory_size))
        
        for t, active_memories in enumerate(attention_history):
            for mem_idx in active_memories:
                if mem_idx < memory_size:
                    heatmap[t, mem_idx] = 1.0
        
        plt.figure(figsize=(12, 6))
        # 転置して (Memory Slot, Time) の向きにする
        plt.imshow(heatmap.T, aspect='auto', cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label="Activation")
        plt.title("Spike Attention Heatmap")
        plt.xlabel("Time Step")
        plt.ylabel("Memory Slot Index")
        plt.tight_layout()
        
        path = f"{self.save_dir}/{filename}"
        plt.savefig(path)
        plt.close()
        print(f"[Viz] Saved attention heatmap to {path}")