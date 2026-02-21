_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_llm_save_load.py",
    "//": "タイトル: スパイキングLLMのモデル保存・読み込みデモ",
    "//": "目的: 学習済みのシナプス結合状態をJSONに保存し、新規インスタンスにロードして推論が再現できるか検証する。"
}

import os
from sara_engine.models.spiking_llm import SpikingLLM

def main():
    print("Starting Model Save/Load Demo...\n")
    
    workspace_dir = os.path.join(os.getcwd(), "workspace", "models")
    model_path = os.path.join(workspace_dir, "spiking_llm_weights.json")
    
    vocab_size = 10000
    vocab_map = {10: "I", 11: "am", 12: "learning", 13: "SNN", 14: "Transformer", 15: ".", 99: "[UNK]"}
    def decode(tokens):
        return " ".join([vocab_map.get(t, str(t)) for t in tokens])

    training_sequence = [10, 11, 12, 13, 14, 15]
    prompt = [10]

    # ==========================================
    # Phase 1: Train and Save
    # ==========================================
    print("[Phase 1: Training original model]")
    model_A = SpikingLLM(vocab_size=vocab_size, d_model=128, num_layers=2, num_heads=4)
    
    for epoch in range(5):
        model_A.learn_sequence(training_sequence)
        
    generated_A = model_A.generate(prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model A (Trained) Generation: {decode(generated_A)}")
    
    print(f"Saving model A state to: {model_path}")
    model_A.save(model_path)
    print("Save complete.\n")

    # ==========================================
    # Phase 2: Load into a brand new model
    # ==========================================
    print("[Phase 2: Loading into a brand new model]")
    model_B = SpikingLLM(vocab_size=vocab_size, d_model=128, num_layers=2, num_heads=4)
    
    # Before loading, verify it generates garbage (untrained state)
    generated_B_before = model_B.generate(prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model B (Before Load) Generation: {decode(generated_B_before)}")
    
    # Load the trained weights
    print("Loading weights...")
    model_B.load(model_path)
    
    # After loading, it should generate exactly what Model A generated
    generated_B_after = model_B.generate(prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model B (After Load) Generation:  {decode(generated_B_after)}")
    
    assert generated_A == generated_B_after, "Error: Loaded model did not reproduce the same output!"
    print("\nSuccess! The learned synaptic weights were perfectly restored.")

if __name__ == "__main__":
    main()