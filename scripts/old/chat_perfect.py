{
    "//": "ディレクトリパス: scripts/chat_perfect.py",
    "//": "ファイルの日本語タイトル: SARA海馬推論スクリプト（BPE完全突破版）",
    "//": "ファイルの目的や内容: 文字列レベルでの青空文庫補正と、BPEの分断を防ぐ「末尾トークン切り捨て検索」を実装し、あらゆる入力から記憶を引き出す。"
}

import msgpack
import time
import os
import numpy as np
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

def run_perfect_chat():
    model_path = "models/distilled_sara_llm.msgpack"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    
    if not os.path.exists(model_path):
        print(f"❌ '{model_path}' が見つかりません。")
        return

    print("Loading Perfect Memory (Hippocampus Engine)...")
    with open(model_path, "rb") as f:
        state = msgpack.unpack(f, raw=False)
    
    direct_map = state.get("direct_map", {})
    print(f"🚀 Successfully loaded {len(direct_map)} pure memories!")

    print("\n" + "="*50)
    print("⚡ SARA Hippocampus Session (BPE-Resilient Mode)")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError): break
        if user_input.strip().lower() in ["quit", "exit"]: break
        if not user_input.strip(): continue

        # 💡 文字列レベルでの検索バリエーション（ここで初めてトークン化する）
        string_variations = [
            user_input,                 # そのまま
            "　" + user_input,           # 青空文庫の段落開始（全角スペース）
            "「" + user_input            # 会話文開始
        ]

        print(f"SARA: ", end="", flush=True)
        start_time = time.time()
        generated_count = 0
        refractory_buffer = []

        # 現在のトークン列（初期状態はNone、検索成功時にセットされる）
        current_tokens = []
        next_id = None

        # 💡 Step 1: 最初の一言目を見つけるための強力な検索
        for text_var in string_variations:
            base_tokens = tokenizer(text_var, return_tensors="pt")["input_ids"][0].tolist()
            
            # BPEの分断対策：「そのまま」と「最後の1トークンを削った状態」の両方を試す
            for drop_last in [False, True]:
                search_tokens = base_tokens[:-1] if drop_last and len(base_tokens) > 2 else base_tokens
                if not search_tokens: continue

                max_window = min(8, len(search_tokens))
                for window in range(max_window, 0, -1):
                    context = search_tokens[-window:]
                    sdr_k = str(student._sdr_key(student._encode_to_sdr(context)))
                    
                    if sdr_k in direct_map:
                        valid_candidates = [
                            (int(cid), w) for cid, w in direct_map[sdr_k].items() 
                        ]
                        
                        if valid_candidates:
                            top_k = min(3, len(valid_candidates))
                            valid_candidates.sort(key=lambda x: x[1], reverse=True)
                            top_candidates = valid_candidates[:top_k]
                            
                            weights = np.array([w for _, w in top_candidates])
                            probs = weights ** 2
                            probs /= probs.sum()
                            
                            chosen_index = np.random.choice(len(top_candidates), p=probs)
                            next_id = top_candidates[chosen_index][0]
                            
                            # 検索成功！ベーストークンを更新
                            current_tokens = search_tokens
                            break
                if next_id is not None: break
            if next_id is not None: break

        # 最初の一言が見つからなかった場合
        if next_id is None:
            print("（その言葉の続きは、まだ漱石の記憶にありません）", flush=True)
            continue

        # 💡 Step 2: 見つかった文脈から言葉を紡ぎ続けるループ
        for step in range(60): 
            if step > 0: # 2単語目以降の検索
                next_id = None
                max_window = min(8, len(current_tokens))
                for window in range(max_window, 0, -1):
                    context = current_tokens[-window:]
                    sdr_k = str(student._sdr_key(student._encode_to_sdr(context)))
                    
                    if sdr_k in direct_map:
                        valid_candidates = [
                            (int(cid), w) for cid, w in direct_map[sdr_k].items() 
                            if int(cid) not in refractory_buffer
                        ]
                        if valid_candidates:
                            top_k = min(3, len(valid_candidates))
                            valid_candidates.sort(key=lambda x: x[1], reverse=True)
                            top_candidates = valid_candidates[:top_k]
                            
                            weights = np.array([w for _, w in top_candidates])
                            probs = weights ** 2
                            probs /= probs.sum()
                            
                            chosen_index = np.random.choice(len(top_candidates), p=probs)
                            next_id = top_candidates[chosen_index][0]
                            break 
                
                if next_id is None:
                    break
            
            # トークンの追加と出力
            current_tokens.append(next_id)
            generated_word = tokenizer.decode([next_id])
            generated_count += 1
            
            print(generated_word, end="", flush=True)
            
            # 直近3単語の不応期ループ防止
            refractory_buffer.append(next_id)
            if len(refractory_buffer) > 3:
                refractory_buffer.pop(0)
            
            # 終了判定
            if generated_word.strip() in ["。", "！", "？", "!", "?", "\n"]:
                break
                
        elapsed_time = time.time() - start_time
        tps = generated_count / elapsed_time if elapsed_time > 0 else 0
        if generated_count > 0:
            print(f"\n      [⏱️ Speed: {tps:.2f} tokens/sec]")

if __name__ == "__main__":
    run_perfect_chat()