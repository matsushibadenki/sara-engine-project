_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_llm_save_load.py",
    "//": "タイトル: スパイキングLLMのモデル保存・読み込みデモ",
    "//": "目的: SpikingLLMのシナプス重みをJSONとして外部ファイルに保存し、新規インスタンスへ復元するロジックを実装する。"
}

import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sara_engine.models.spiking_llm import SpikingLLM

def save_spiking_llm(model, filepath):
    """モデル内の重み辞書をJSONシリアライズして保存"""
    state = {
        "lm_head_w": model.lm_head_w,
        "transformer": [{"ffn_w": layer.ffn_w} for layer in model.transformer.layers]
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(state, f)

def load_spiking_llm(model, filepath):
    """JSONから重みを読み込み、文字列キーをintに変換して復元"""
    with open(filepath, 'r', encoding="utf-8") as f:
        state = json.load(f)
    model.lm_head_w = [{int(k): float(v) for k, v in layer.items()} for layer in state["lm_head_w"]]
    for i, layer_state in enumerate(state["transformer"]):
        model.transformer.layers[i].ffn_w = [{int(k): float(v) for k, v in fw.items()} for fw in layer_state["ffn_w"]]

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
    model_A = SpikingLLM(vocab_size=vocab_size, d_model=128, num_layers=2)
    
    for epoch in range(5):
        model_A.learn_sequence(training_sequence)
        
    generated_A = model_A.generate(prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model A (Trained) Generation: {decode(generated_A)}")
    
    print(f"Saving model A state to: {model_path}")
    save_spiking_llm(model_A, model_path)
    print("Save complete.\n")

    # ==========================================
    # Phase 2: Load into a brand new model
    # ==========================================
    print("[Phase 2: Loading into a brand new model]")
    model_B = SpikingLLM(vocab_size=vocab_size, d_model=128, num_layers=2)
    
    generated_B_before = model_B.generate(prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model B (Before Load) Generation: {decode(generated_B_before)}")
    
    print("Loading weights...")
    load_spiking_llm(model_B, model_path)
    
    generated_B_after = model_B.generate(prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model B (After Load) Generation:  {decode(generated_B_after)}")
    
    assert generated_A == generated_B_after, "Error: Loaded model did not reproduce the same output!"
    print("\nSuccess! The learned synaptic weights were perfectly restored.")

if __name__ == "__main__":
    main()