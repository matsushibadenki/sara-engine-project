"""
{
    "//": "ディレクトリパス: src/sara_engine/utils/sara_board.py",
    "//": "ファイルの日本語タイトル: Sara-Board (TensorBoard代替のSNN可視化ツール)",
    "//": "ファイルの目的や内容: スパイク発火のラスタープロットやマクロな脳波(LFP)、動的しきい値の変動を記録し、画像として出力する軽量な可視化ユーティリティ。"
}
"""

import os
import json
from typing import List, Dict, Any

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class SaraBoard:
    """
    Lightweight visualization and logging tool for SARA Engine.
    Tracks neural dynamics like spikes (raster plot), LFP (Local Field Potential), 
    and homeostatic thresholds over time.
    """
    def __init__(self, run_name: str = "default_run"):
        self.run_name = run_name
        # log_data format: { "layer_name": { "spikes": [...], "thresholds": [...], "lfp": [...] } }
        self.log_data: Dict[str, Dict[str, List[Any]]] = {}

    def log_spikes(self, layer_name: str, spikes: List[List[bool]]):
        """Logs a sequence of spikes for a raster plot and LFP calculation."""
        if layer_name not in self.log_data:
            self.log_data[layer_name] = {"spikes": [], "thresholds": [], "lfp": []}
            
        for step_spikes in spikes:
            # LFP is roughly the sum of synchronous firing activity (macro potential)
            lfp_val = sum(1.0 for s in step_spikes if s)
            self.log_data[layer_name]["spikes"].append(step_spikes)
            self.log_data[layer_name]["lfp"].append(lfp_val)

    def log_thresholds(self, layer_name: str, thresholds: List[float]):
        """Logs the average dynamic thresholds (Homeostasis)."""
        if layer_name not in self.log_data:
            self.log_data[layer_name] = {"spikes": [], "thresholds": [], "lfp": []}
        
        avg_thresh = sum(thresholds) / max(1, len(thresholds))
        self.log_data[layer_name]["thresholds"].append(avg_thresh)

    def save_logs_json(self, filepath: str):
        """Saves raw log data to JSON (if matplotlib is unavailable)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2)

    def plot_and_save(self, save_dir: str):
        """Plots the recorded dynamics and saves to PNG files."""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib is not installed. Saving logs as JSON instead.")
            self.save_logs_json(os.path.join(save_dir, f"{self.run_name}_logs.json"))
            return

        os.makedirs(save_dir, exist_ok=True)

        for layer_name, data in self.log_data.items():
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"Sara-Board Dynamics: {layer_name} ({self.run_name})", fontsize=14)

            time_steps = len(data["spikes"])
            if time_steps == 0:
                continue

            # 1. Raster Plot (Micro-dynamics)
            neuron_indices = []
            spike_times = []
            num_neurons = len(data["spikes"][0])
            
            for t, step_spikes in enumerate(data["spikes"]):
                for n_idx, spiked in enumerate(step_spikes):
                    if spiked:
                        spike_times.append(t)
                        neuron_indices.append(n_idx)

            axs[0].scatter(spike_times, neuron_indices, s=2, color='black', marker='|')
            axs[0].set_ylabel("Neuron ID")
            axs[0].set_title("Spike Raster Plot")
            axs[0].set_ylim(-1, num_neurons)

            # 2. Local Field Potential (Macro-dynamics)
            axs[1].plot(range(time_steps), data["lfp"], color='blue', linewidth=1.5)
            axs[1].set_ylabel("Activity (Spikes)")
            axs[1].set_title("Local Field Potential (LFP)")

            # 3. Average Homeostatic Threshold
            if data["thresholds"]:
                axs[2].plot(range(len(data["thresholds"])), data["thresholds"], color='red', linewidth=1.5)
                axs[2].set_ylabel("Avg Threshold")
                axs[2].set_title("Homeostasis (Dynamic Threshold)")
            
            axs[2].set_xlabel("Time Step")
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{self.run_name}_{layer_name}_dynamics.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[Sara-Board] Saved visualization to: {save_path}")