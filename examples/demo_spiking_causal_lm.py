_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_causal_lm.py",
    "//": "ファイルの日本語タイトル: スパイキング因果言語モデルのデモ",
    "//": "ファイルの目的や内容: SNNトランスフォーマーを使った言語モデルの学習とテキスト生成テスト。行列演算を用いずに文脈からのトークン予測が可能か検証する。"
}

import os
import time
from sara_engine.models.spiking_causal_lm import SpikingCausalLM

def main():
    print("Starting Spiking Causal LM Demo...")
    
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Initialize the model (vocab size assumes a small character or word level mapping for demo)
    model = SpikingCausalLM(vocab_size=1000, embed_dim=1024, hidden_dim=2048, use_lif=True)
    
    # Mock Token Sequences (e.g., words mapped to integer IDs)
    # 1: "Hello", 2: "SARA", 3: "Engine", 4: "is", 5: "fast"
    seq_a = [1, 2, 3, 4, 5] 
    # 6: "The", 7: "AI", 4: "is", 8: "lightweight"
    seq_b = [6, 7, 4, 8]
    
    print("\n--- Training (Hebbian & STDP) ---")
    start_time = time.time()
    
    # Train sequentially
    for epoch in range(3):  # 3 epochs to strengthen synapses
        model.train_step(seq_a)
        model.train_step(seq_b)
        print(f"Epoch {epoch+1} completed.")
        
    print(f"Training completed in {time.time() - start_time:.4f} seconds.")
    
    # Save Model
    save_path = os.path.join(workspace_dir, "spiking_causal_lm.json")
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    print("\n--- Text Generation (Inference) ---")
    
    # Test 1: Prompt "Hello SARA" -> expects "Engine is fast"
    prompt_1 = [1, 2]
    print(f"Prompt 1: {prompt_1} -> Generating...")
    generated_1 = model.generate(prompt_1, max_new_tokens=3, temperature=0.1) # Greedy
    print(f"Generated 1: {generated_1}")
    
    # Test 2: Prompt "The AI" -> expects "is lightweight"
    prompt_2 = [6, 7]
    print(f"Prompt 2: {prompt_2} -> Generating...")
    generated_2 = model.generate(prompt_2, max_new_tokens=2, temperature=0.1)
    print(f"Generated 2: {generated_2}")

if __name__ == "__main__":
    main()