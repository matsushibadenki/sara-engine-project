# [配置するディレクトリのパス]: ./scripts/old/chat_rust_core.py
# [ファイルの日本語タイトル]: Rustコア推論スクリプト (ソフト・ペナルティ修正版)
# [ファイルの目的や内容]: 8192ニューロンのまま、助詞のハブ化のみを安全な対数ペナルティで抑制し、ワードサラダを完全に防ぐ。
{
    "//": "ディレクトリパス: scripts/chat_rust_core.py",
    "//": "ファイルの日本語タイトル: Rustコア推論スクリプト (ソフト・ペナルティ修正版)",
    "//": "ファイルの目的や内容: 8192ニューロンのまま、助詞のハブ化のみを安全な対数ペナルティで抑制し、ワードサラダを完全に防ぐ。"
}

import time
import os
import math
import tqdm
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

try:
    from sara_engine import sara_rust_core # type: ignore
except ImportError:
    print("❌ sara_rust_core が見つかりません。")
    exit(1)

def run_rust_chat():
    model_path = "models/distilled_sara_llm.msgpack"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    sdr_size = 8192 
    student = SpikingLLM(num_layers=2, sdr_size=sdr_size, vocab_size=256000)
    rust_engine = sara_rust_core.SpikeEngine()
    
    if not os.path.exists(model_path):
        print(f"❌ Error: '{model_path}' が見つかりません。")
        return

    print(f"Loading distilled knowledge from {model_path}...")
    student.load_memory(model_path)
    direct_map = student._direct_map
    items = list(direct_map.items())
    
    print("Analyzing neural pathways (Applying Safe Soft-Penalty)...")
    token_freq: dict[int, int] = {}
    for _, next_tokens in items:
        for tok_id, count in next_tokens.items():
            token_freq[tok_id] = token_freq.get(tok_id, 0) + 1

    weights: list[dict[int, float]] = [{} for _ in range(sdr_size)]
    
    for sdr_k, next_tokens in tqdm.tqdm(items, desc="Transferring to Rust Core"):
        for tok_id, count in next_tokens.items():
            freq = token_freq.get(tok_id, 1)
            
            # 💡 修正点：希少な言葉を爆発させず、10回以上出現した言葉のみを対数で優しく抑制
            penalty = 1.0
            if freq > 10:
                penalty = 1.0 / math.log10(freq)
            
            weight_per_spike = (float(count) / len(sdr_k)) * penalty
            
            for pre_id in sdr_k:
                weights[pre_id][tok_id] = max(weights[pre_id].get(tok_id, 0.0), weight_per_spike)

    rust_engine.set_weights(weights)
    print(f"🚀 Successfully transferred {len(items)} patterns into Rust Core!")

    print("\n" + "="*50)
    print("⚡ SARA Rust Core Session (Safe Hub-Suppression)")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError): break
        if user_input.strip().lower() in ["quit", "exit"]: break
        if not user_input.strip(): continue

        inputs = tokenizer(user_input, return_tensors="pt")
        current_tokens = inputs["input_ids"][0].tolist()

        print(f"SARA: ", end="", flush=True)
        start_time = time.time()
        generated_count = 0
        refractory_buffer = []

        # 💡 スパイクが正常化されたため、適切な閾値を設定
        fire_threshold = 40.0 

        for step in range(50): 
            context_tokens = current_tokens[-8:]
            sdr = student._encode_to_sdr(context_tokens)
            
            # 候補を5つ取得
            out_spikes = rust_engine.propagate(sdr, fire_threshold, 5)
            
            if not out_spikes:
                if step == 0:
                    print("（まだ学習していない言葉の繋がりです）", end="")
                break
                
            next_id = None
            for candidate in out_spikes:
                if candidate not in refractory_buffer:
                    next_id = candidate
                    break
            
            if next_id is None:
                next_id = out_spikes[0]
                
            current_tokens.append(next_id)
            generated_word = tokenizer.decode([next_id])
            generated_count += 1
            print(generated_word, end="", flush=True)
            
            refractory_buffer.append(next_id)
            if len(refractory_buffer) > 4:
                refractory_buffer.pop(0)
            
            if generated_word.strip() in ["。", "！", "？", "!", "?", "\n"]:
                break
                
        elapsed_time = time.time() - start_time
        tps = generated_count / elapsed_time if elapsed_time > 0 else 0
        if generated_count > 0:
            print(f"\n      [⏱️ Speed: {tps:.2f} tokens/sec]")

if __name__ == "__main__":
    run_rust_chat()
