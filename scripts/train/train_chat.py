# ディレクトリパス: scripts/train_chat.py
# ファイルの日本語タイトル: 対話データ蒸留スクリプト
# ファイルの目的や内容: 複数のソースからチャットデータを読み込み、SNN（modelsディレクトリ）に記憶させる。

import torch
import msgpack
import os
import json
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

def train_chat_data(data_paths, model_path):
    print("Initializing SNN Student Model (8192 neurons)...")
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    device = "cpu"
    
    print(f"Loading teacher model: google/gemma-2-2b on {device}")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    teacher = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.float32, device_map=device)
    teacher.eval()
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Opening SNN memory file: {model_path}...")
        with open(model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        raw_map = state.get("direct_map", {})
        student._direct_map = {eval(k): {int(tk): float(tv) for tk, tv in v.items()} for k, v in raw_map.items()}
        print(f"✅ Loaded {len(student._direct_map)} patterns.")
    else:
        print("⚠️ 既存の記憶がありません。新規作成します。")
        student._direct_map = {}

    chat_lines = []
    for dp in data_paths:
        if os.path.exists(dp):
            print(f"読み込み中: {dp}")
            with open(dp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "user" in item and "sara" in item:
                            chat_lines.append(f"You: {item['user']}\nSARA: {item['sara']}\n")
                        elif "text" in item:
                            text = item["text"].replace("ユーザー:", "You:").replace("システム:", "SARA:") + "\n"
                            chat_lines.append(text)
        else:
            print(f"⚠️ '{dp}' が見つかりません。")
                
    if not chat_lines:
        print("❌ 学習できるデータがありません。")
        return

    print(f"🚀 {len(chat_lines)}件の対話データを学習します...")
    
    for text in tqdm.tqdm(chat_lines, desc="Chat Training"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"][0].tolist()
        if len(input_ids) < 2: continue

        with torch.no_grad():
            outputs = teacher(**inputs)
            probs = torch.softmax(outputs.logits[0], dim=-1)

        context_tokens = []
        for i in range(len(input_ids) - 1):
            context_tokens.append(input_ids[i])
            if len(context_tokens) > 24: context_tokens.pop(0)
            
            sdr_k = student._sdr_key(student._encode_to_sdr(context_tokens))
            if sdr_k not in student._direct_map: student._direct_map[sdr_k] = {}
            dm = student._direct_map[sdr_k]
            actual = input_ids[i+1]
            
            dm[actual] = dm.get(actual, 0.0) + 500.0
            
            top_probs, top_indices = torch.topk(probs[i], 5)
            for rank in range(5):
                t_idx = top_indices[rank].item()
                if t_idx != actual:
                    dm[t_idx] = dm.get(t_idx, 0.0) + 50.0 * top_probs[rank].item()
                    
            for tok_id in list(dm.keys()):
                if dm[tok_id] > 2000.0: dm[tok_id] = 2000.0

    print("Saving updated memory...")
    state = {
        "direct_map": {str(k): {str(tk): v for tk, v in tv.items()} for k, tv in student._direct_map.items()},
        "vocab_size": student.vocab_size
    }
    with open(model_path, "wb") as f:
        msgpack.pack(state, f)
    print("✨ 対話学習が完了しました！")