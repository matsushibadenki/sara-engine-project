_FILE_INFO = {
    "path": "examples/demo_spiking_transformer.py",
    "title": "スパイキング・トランスフォーマー・デモ",
    "description": "SNN版Transformerブロックの動作確認とSTDPによる学習のデモンストレーション。"
}

import os
import json
import random
from sara_engine.core.transformer import SpikeTransformerBlock

def main():
    print("Starting Spiking Transformer Demo...")
    
    # Ensure workspace directory exists for outputs (to prevent project bloating)
    workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "spiking_transformer_log.json")
    
    embed_dim = 256
    hidden_dim = 512
    seq_length = 20
    
    # Initialize the Spike Transformer Block
    block = SpikeTransformerBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, density=0.1, context_size=64)
    
    # Generate dummy multilingual input spikes (e.g., token encoding simulation)
    # Independent of language structure, simply simulated as active network nodes
    input_sequence = []
    for _ in range(seq_length):
        # Base input firing rate of ~10%
        spikes = random.sample(range(embed_dim), int(embed_dim * 0.1))
        input_sequence.append(spikes)
        
    logs = []
    
    print("Running forward pass with STDP learning...")
    for step, x_spikes in enumerate(input_sequence):
        # Forward pass with learning enabled (Local STDP active)
        y_spikes = block.forward(x_spikes, learning=True)
        
        log_entry = {
            "step": step,
            "input_spike_count": len(x_spikes),
            "output_spike_count": len(y_spikes)
        }
        logs.append(log_entry)
        print(f"Step {step}: Input Spikes = {len(x_spikes)}, Output Spikes = {len(y_spikes)}")
        
    # Save test logs to workspace
    with open(log_file_path, "w") as f:
        json.dump(logs, f, indent=4)
        
    print(f"Demo completed. Logs saved to {log_file_path}")

if __name__ == "__main__":
    main()