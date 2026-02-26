{
    "//": "ディレクトリパス: scripts/chat_rust_core.py",
    "//": "ファイルの日本語タイトル: Rustコア搭載・超高速チャットスクリプト（SQLite & パス最適化版）",
    "//": "ファイルの目的や内容: models/ 以下のMessagePackモデルを読み込み、Rustエンジンで推論を行う。閾値制御により未知の話題には「わかりません」と応答する機能を搭載。"
}

import msgpack
import time
import os
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

try:
    from sara_engine import sara_rust_core
except ImportError:
    print("❌ sara_rust_core が見つかりません。Rustモジュールをビルドしてください。")
    exit(1)

def run_rust_chat():
    # 💡 パスの設定（ルートからの相対パス）
    model_path = "models/distilled_sara_llm.msgpack"
    
    print("Loading tokenizer and initializing models...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    # SNNの基本パラメータ（蒸留時と一致させる）
    sdr_size = 8192 
    student = SpikingLLM(num_layers=2, sdr_size=sdr_size, vocab_size=256000)
    
    rust_engine = sara_rust_core.SpikeEngine()
    
    if not os.path.exists(model_path):
        print(f"❌ Error: '{model_path}' が見つかりません。先に蒸留を実行してください。")
        return

    print(f"Loading distilled knowledge from {model_path} into Rust Core...")
    try:
        with open(model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
            
        weights = [{} for _ in range(sdr_size)]
        pattern_count = 0
        
        # MessagePackから重みを展開
        for str_sdr_k, next_tokens in state["direct_map"].items():
            sdr_k = eval(str_sdr_k) 
            pattern_count += 1
            
            for str_tok_id, count in next_tokens.items():
                tok_id = int(str_tok_id)
                # スパイクあたりの重みを計算
                weight_per_spike = float(count) / len(sdr_k) 
                
                for pre_id in sdr_k:
                    # 既存の重みと最大値を比較して保持（ハブノード抑制）
                    weights[pre_id][tok_id] = max(weights[pre_id].get(tok_id, 0.0), weight_per_spike)

        rust_engine.set_weights(weights)
        print(f"🚀 Successfully transferred {pattern_count} patterns into Rust Core!")
        del state # メモリ解放
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        return

    print("\n" + "="*50)
    print("⚡ SARA Rust Core Session Started (Multi-core Optimized)")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
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

        # 💡 厳格な発火閾値（誤発火防止）
        fire_threshold = 60.0 

        for step in range(30):
            # 直近8トークンの文脈を使用
            context_tokens = current_tokens[-8:]
            sdr = student._encode_to_sdr(context_tokens)
            
            # Rustコアでスパイク伝播
            out_spikes = rust_engine.propagate(sdr, fire_threshold, 10)
            
            # 閾値を超える候補がない場合
            if not out_spikes:
                if step == 0:
                    print("すみません、その話題についてはまだ学習していません。", end="")
                break
                
            # 不応期（Refractory Period）チェック: 直近3単語のループを防止
            next_id = None
            for candidate in out_spikes:
                if candidate not in refractory_buffer:
                    next_id = candidate
                    break
            
            # 候補がすべて不応期ならトップを採用
            if next_id is None:
                next_id = out_spikes[0]
                
            current_tokens.append(next_id)
            generated_word = tokenizer.decode([next_id])
            generated_count += 1
            
            print(generated_word, end="", flush=True)
            
            # 不応期バッファの更新
            refractory_buffer.append(next_id)
            if len(refractory_buffer) > 3:
                refractory_buffer.pop(0)
            
            # 終端記号で生成を終了
            if generated_word.strip() in ["。", "！", "？", "!", "?", "\n"]:
                break
                
        elapsed_time = time.time() - start_time
        tps = generated_count / elapsed_time if elapsed_time > 0 else 0
        
        if generated_count > 0:
            print(f"\n      [⏱️ Speed: {tps:.2f} tokens/sec]")
        else:
            print()

if __name__ == "__main__":
    run_rust_chat()