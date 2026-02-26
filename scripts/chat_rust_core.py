# ディレクトリパス: scripts/chat_rust_core.py
# ファイルの日本語タイトル: Rustコア搭載・超高速チャットスクリプト（未知語フォールバック対応版）
# ファイルの目的や内容: SNNの確信度（発火閾値）を利用し、未知の文脈が入力された際にノイズを出力するのではなく「わかりません」と返す機能を実装。

FILE_INFO = {
    "//": "コメント: 閾値制御によりハルシネーション（知ったかぶり）を防止します。"
}

import json
import time
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

try:
    from sara_engine import sara_rust_core
except ImportError:
    print("❌ sara_rust_core が見つかりません。")
    exit(1)

def run_rust_chat():
    print("Loading tokenizer and initializing models...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    sdr_size = 8192 
    student = SpikingLLM(num_layers=2, sdr_size=sdr_size, vocab_size=256000)
    
    rust_engine = sara_rust_core.SpikeEngine()
    
    print("Loading distilled knowledge into Rust Core...")
    try:
        # 拡張子がmsgpackの場合は msgpack モジュールが必要ですが、
        # 直近のテストに合わせてMessagePackからの読み込み処理を組み込みます。
        import msgpack
        import os
        model_path = "distilled_sara_llm.msgpack"
        
        if not os.path.exists(model_path):
            print(f"❌ Error: '{model_path}' が見つかりません。")
            return
            
        with open(model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
            
        weights = [{} for _ in range(sdr_size)]
        pattern_count = 0
        
        for str_sdr_k, next_tokens in state["direct_map"].items():
            sdr_k = eval(str_sdr_k) 
            pattern_count += 1
            
            for str_tok_id, count in next_tokens.items():
                tok_id = int(str_tok_id)
                weight_per_spike = float(count) / len(sdr_k) 
                
                for pre_id in sdr_k:
                    if tok_id not in weights[pre_id]:
                        weights[pre_id][tok_id] = 0.0
                    
                    weights[pre_id][tok_id] = max(weights[pre_id][tok_id], weight_per_spike)

        rust_engine.set_weights(weights)
        print(f"🚀 Successfully transferred {pattern_count} patterns into Rust Core!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("\n" + "="*50)
    print("⚡ SARA Rust Core Session Started (MessagePack & Fallback Optimized)")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue

        inputs = tokenizer(user_input, return_tensors="pt")
        current_tokens = inputs["input_ids"][0].tolist()

        print(f"SARA: ", end="", flush=True)
        
        start_time = time.time()
        generated_count = 0
        refractory_buffer = []

        # 💡 発火閾値の設定（この数値以下の確信度の単語は無視する）
        # Teacher Forcing で 100.0 などの重みをつけているため、
        # 完全に未知の文脈だと合計値がこれを下回ります。環境に合わせて調整可能です。
        fire_threshold = 2.0 

        for step in range(30):
            context_tokens = current_tokens[-8:]
            sdr = student._encode_to_sdr(context_tokens)
            
            # 💡 閾値を設定してRustコアを呼び出し
            out_spikes = rust_engine.propagate(sdr, fire_threshold, 10)
            
            # 💡 閾値を超える単語が1つも見つからなかった場合の処理
            if not out_spikes:
                if step == 0:
                    # 1文字も生成できずに終わった＝完全に知らない話題
                    print("すみません、その話題についてはまだ学習していません。", end="")
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
            if len(refractory_buffer) > 3:
                refractory_buffer.pop(0)
            
            if generated_word.strip() in ["。", "！", "？", "!", "?", "\n"]:
                break
                
        elapsed_time = time.time() - start_time
        tps = generated_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n      [⏱️ Speed: {tps:.2f} tokens/sec]")

if __name__ == "__main__":
    run_rust_chat()