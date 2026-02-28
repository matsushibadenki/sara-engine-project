_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_llm.py",
    "//": "タイトル: スパイキング・トランスフォーマー実行デモ",
    "//": "目的: 新しいSpikeTokenizerの仕様に合わせて初期化とエンコード処理を修正し、トランスフォーマーを正常に動作させる。"
}

import os
import sys
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from src.sara_engine.models.spiking_llm import SpikingTransformerBlock

def main():
    print("Initializing SARA Spiking Transformer...")
    sdr_size = 1024
    
    # 新しいSpikeTokenizerにはsdr_size等の引数は不要
    tokenizer = SpikeTokenizer()
    transformer_block = SpikingTransformerBlock(sdr_size=sdr_size)
    
    input_text = "Hello SNN! こんにちは、世界。Bonjour!"
    print(f"Input Text: {input_text}")
    
    # トークナイザーの語彙を学習させてからエンコードする
    tokenizer.train([input_text])
    token_ids = tokenizer.encode(input_text)
    
    output_log = []
    
    print("Starting temporal simulation...")
    for t_step, token_id in enumerate(token_ids):
        # トークンIDをスパイク（発火ニューロンのインデックス）に変換
        spikes = [token_id % sdr_size]
        out_spikes = transformer_block.forward(spikes, t_step=t_step)
        
        log_entry = {
            "time_step": t_step,
            "token_id": token_id,
            "input_spikes_count": len(spikes),
            "output_spikes_count": len(out_spikes),
            "active_ratio": round(len(out_spikes) / sdr_size, 4)
        }
        output_log.append(log_entry)
        print(f"Time: {t_step:3d} | Token: {token_id:3d} | Input Spikes: {len(spikes):3d} | Output Spikes: {len(out_spikes):3d}")

    workspace_dir = os.path.join(project_root, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    log_file_path = os.path.join(workspace_dir, "spiking_transformer_log.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(output_log, f, indent=4)
        
    print(f"Simulation complete. Log saved to {log_file_path}")

if __name__ == "__main__":
    main()