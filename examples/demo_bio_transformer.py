# // examples/demo_bio_transformer.py
# // 生物学的Transformerのデモスクリプト
# // 目的や内容: 多言語テキストをバイナリスパイクにエンコードし、BioTransformerに入力してSTDP学習と発火結果のログ・画像出力を実行します。実行中にコンソールへ進捗を出力し、動作状況を可視化しています。

import os
import sys
import matplotlib.pyplot as plt

# Add the project root directory to sys.path to resolve the 'src' module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sara_engine.models.bio_transformer import BioSpikingTransformer

def text_to_spikes(text: str, seq_len: int, d_model: int) -> list[list[int]]:
    encoded = text.encode('utf-8')
    spikes = [[0 for _ in range(d_model)] for _ in range(seq_len)]
    
    for i in range(min(seq_len, len(encoded))):
        byte_val = encoded[i]
        for j in range(8):
            if d_model > j:
                spikes[i][j] = (byte_val >> j) & 1
        
        for j in range(8, d_model):
            spikes[i][j] = (byte_val * (j + 1)) % 2
            
    return spikes

def run_demo():
    workspace_dir = os.path.join(os.getcwd(), "workspace", "bio_transformer")
    os.makedirs(workspace_dir, exist_ok=True)
    
    log_file_path = os.path.join(workspace_dir, "training_log.txt")
    
    seq_len = 16
    d_model = 16
    d_ff = 32
    num_layers = 2
    timesteps = 50

    print(f"Initializing BioSpikingTransformer...")
    print(f"Layers: {num_layers}, Seq_len: {seq_len}, D_model: {d_model}")
    model = BioSpikingTransformer(num_layers, seq_len, d_model, d_ff)
    
    multilingual_text = "Hello世界! SNN" 
    print(f"Encoding text: '{multilingual_text}' to binary spikes...")
    base_spikes = text_to_spikes(multilingual_text, seq_len, d_model)

    output_history = []
    
    print(f"\nStarting simulation for {timesteps} timesteps.")
    print(f"Detailed logs will be saved to: {log_file_path}\n")
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("Starting biological transformer training (STDP only, No Backprop, No MatMul)...\n")
        f.write(f"Input text: {multilingual_text}\n\n")
        
        for t in range(timesteps):
            out_spikes = model.forward(base_spikes, t)
            
            total_spikes = sum(sum(token) for token in out_spikes)
            log_msg = f"Timestep {t:02d}: Generated {total_spikes} spikes across all sequences."
            
            f.write(log_msg + "\n")
            
            # コンソールには多すぎないよう10ステップごと、および初回と最後に表示
            if t % 10 == 0 or t == timesteps - 1:
                print(log_msg)
            
            flat_out = []
            for i in range(seq_len):
                for d in range(d_model):
                    if out_spikes[i][d] > 0:
                        flat_out.append((t, i * d_model + d))
            output_history.extend(flat_out)
            
        f.write("\nTraining complete. Artifacts saved in workspace.")
    
    print("\nSimulation complete. Generating raster plot...")
    
    if output_history:
        times, neurons = zip(*output_history)
        plt.figure(figsize=(10, 6))
        plt.scatter(times, neurons, s=5, c='black', marker='.')
        plt.title("Output Spike Raster Plot")
        plt.xlabel("Timestep")
        plt.ylabel("Neuron Index")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        img_path = os.path.join(workspace_dir, "raster_plot.png")
        plt.savefig(img_path)
        plt.close()
        print(f"Raster plot successfully saved to: {img_path}")
    else:
        print("No spikes were generated to plot.")

if __name__ == "__main__":
    run_demo()