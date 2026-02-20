# filepath: examples/demo_snn_llm.py
# title: スパイキング・トランスフォーマー実行デモ
# description: 多言語テキストを入力としてSNNベースのTransformerモジュールを動作させ、発火状態の結果をworkspaceディレクトリに出力する。

import os
import sys
import json

# プロジェクトルートディレクトリをsys.pathに追加して、srcモジュールを読み込めるようにする
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from src.sara_engine.models.spiking_llm import SpikingTransformerBlock

def main():
    print("Initializing SARA Spiking Transformer...")
    sdr_size = 1024
    tokenizer = SpikeTokenizer(sdr_size=sdr_size, active_bits=32)
    transformer_block = SpikingTransformerBlock(sdr_size=sdr_size)
    
    # Multilingual text input
    input_text = "Hello SNN! こんにちは、世界。Bonjour!"
    print(f"Input Text: {input_text}")
    
    # Encode text to temporal spike trains
    spike_trains = tokenizer.encode(input_text, time_window=10)
    
    output_log = []
    
    print("Starting temporal simulation...")
    for t_step, spikes in spike_trains:
        # Feed into transformer block
        out_spikes = transformer_block.forward(spikes)
        
        log_entry = {
            "time_step": t_step,
            "input_spikes_count": len(spikes),
            "output_spikes_count": len(out_spikes),
            "active_ratio": round(len(out_spikes) / sdr_size, 4)
        }
        output_log.append(log_entry)
        print(f"Time: {t_step:3d} | Input Spikes: {len(spikes):3d} | Output Spikes: {len(out_spikes):3d}")

    # Ensure workspace directory exists at the root level
    workspace_dir = os.path.join(project_root, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    log_file_path = os.path.join(workspace_dir, "spiking_transformer_log.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(output_log, f, indent=4)
        
    print(f"Simulation complete. Log saved to {log_file_path}")

if __name__ == "__main__":
    main()