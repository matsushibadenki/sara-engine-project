_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_lif_transformer.py",
    "//": "ファイルの日本語タイトル: LIFトランスフォーマーのデモ",
    "//": "ファイルの目的や内容: LIF（Leaky Integrate-and-Fire）モデルを用いたSNNトランスフォーマーの動作確認。長文の文脈理解をシミュレートし、成果物はworkspace下に出力する。"
}

import os
import time
from sara_engine.core.transformer import SpikeTransformerModel

def main():
    print("Starting LIF Spike Transformer Demo...")
    
    # Create workspace directory for outputs
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Initialize the model with LIF mode enabled
    model = SpikeTransformerModel(
        num_layers=2,
        embed_dim=1024,
        hidden_dim=2048,
        use_lif=True
    )
    
    # Simulated sequence of tokens represented by sparse spikes
    sequence = [
        [10, 20, 30],       # Token 1
        [15, 25],           # Token 2
        [10, 40],           # Token 3
        [50, 60, 70],       # Token 4
        [20, 80]            # Token 5
    ]
    
    print("\n--- Training / Context Building ---")
    model.reset_state()
    start_time = time.time()
    
    for i, spikes in enumerate(sequence):
        # Sequentially feed spikes to build up membrane potential (context)
        out_spikes = model.forward(spikes, learning=True)
        print(f"Step {i+1}: Input {spikes} -> Output Spikes: {len(out_spikes)}")
        
    print(f"Training completed in {time.time() - start_time:.4f} seconds.")
    
    # Save the model
    save_path = os.path.join(workspace_dir, "lif_transformer.json")
    model.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")
    
    # Inference mode (testing context retention)
    print("\n--- Inference ---")
    model.reset_state()
    
    # Feed earlier context
    model.forward(sequence[0], learning=False)
    model.forward(sequence[1], learning=False)
    
    # Next prediction based on retained membrane potentials
    pred_spikes = model.forward(sequence[2], learning=False)
    print(f"Inference given sequence[0:3] -> Output Spikes: {pred_spikes[:10]}... (Total: {len(pred_spikes)})")

if __name__ == "__main__":
    main()