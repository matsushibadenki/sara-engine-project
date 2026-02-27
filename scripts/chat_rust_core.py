# ディレクトリパス: scripts/chat_rust_core.py
# ファイルの日本語タイトル: Rustコア推論スクリプト (クリーン版)
# ファイルの目的や内容: 正常な脳に対応した、無駄なフィルターのない最速の推論コード。

import msgpack
import time
import os
import tqdm
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

try:
    from sara_engine import sara_rust_core
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
    try:
        with open(model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        
        weights = [{} for _ in range(sdr_size)]
        
        items = state.get("direct_map", {}).items()
        for str_sdr_k, next_tokens in tqdm.tqdm(items, desc="Transferring to Rust Core"):
            sdr_k = eval(str_sdr_k)
            for str_tok_id, count in next_tokens.items():
                tok_id = int(str_tok_id)
                weight_per_spike = float(count) / len(sdr_k)
                for pre_id in sdr_k:
                    weights[pre_id][tok_id] = max(weights[pre_id].get(tok_id, 0.0), weight_per_spike)

        rust_engine.set_weights(weights)
        print(f"🚀 Successfully transferred {len(items)} patterns!")
        del state
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print("\n" + "="*50)
    print("⚡ SARA Rust Core Session (Clean State)")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError): break
        if user_input.strip().lower() in ["quit", "exit"]: break
        if not user_input.strip(): continue

        inputs = tokenizer(user_input, return_tensors="pt")
        current_tokens = inputs["input_ids"][0].tolist()

        print(f"SARA: ", end="", flush=True)
        start_time = time.time()
        generated_count = 0
        refractory_buffer = []

        # 💡 通常の閾値に戻す
        fire_threshold = 60.0 

        for step in range(50): 
            context_tokens = current_tokens[-8:]
            sdr = student._encode_to_sdr(context_tokens)
            
            # 💡 シンプルにトップの候補をもらう
            out_spikes = rust_engine.propagate(sdr, fire_threshold, 3)
            
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
        print(f"\n      [⏱️ Speed: {tps:.2f} tokens/sec]")

if __name__ == "__main__":
    run_rust_chat()