_FILE_INFO = {
    "//": "ディレクトリパス: scripts/train_chat.py",
    "//": "ファイルの日本語タイトル: 対話データ蒸留スクリプト",
    "//": "ファイルの目的や内容: 煩雑なデータ管理を避けるため、JSONLから対話パターンのみを独立してSNNに学習させる。"
}

import torch
import msgpack
import os
import json
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

def train_chat_data():
    model_path = "models/distilled_sara_llm.msgpack"
    data_path = "data/chat_data.jsonl"
    
    if not os.path.exists(data_path):
        print(f"❌ '{data_path}' が見つかりません。")
        return
        
    print("Initializing SNN Student Model (8192 neurons)...")
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading teacher model: google/gemma-2-2b on {device}")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    teacher = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", 
        torch_dtype=torch.float32, 
        device_map=device
    )
    teacher.eval()
    
    if os.path.exists(model_path):
        print(f"Opening SNN memory file: {model_path}...")
        with open(model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        
        raw_map = state.get("direct_map", {})
        fixed_map = {}
        for k, v in raw_map.items():
            fixed_map[eval(k)] = {int(tk): float(tv) for tk, tv in v.items()}
        student._direct_map = fixed_map
        print(f"✅ Loaded {len(fixed_map)} patterns.")
    else:
        print("⚠️ 既存の記憶がありません。")
        student._direct_map = {}

    chat_lines = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chat_lines.append(json.loads(line))
                
    print(f"🚀 {len(chat_lines)}件の対話データを学習します...")
    
    for item in tqdm.tqdm(chat_lines, desc="Chat Training"):
        # 💡 SARAに「会話の型」を教え込むフォーマット
        text = f"You: {item['user']}\nSARA: {item['sara']}\n"
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"][0].tolist()
        if len(input_ids) < 2: continue

        with torch.no_grad():
            outputs = teacher(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        context_tokens = []
        for i in range(len(input_ids) - 1):
            context_tokens.append(input_ids[i])
            if len(context_tokens) > 8: context_tokens.pop(0)
            
            sdr_k = student._sdr_key(student._encode_to_sdr(context_tokens))
            if sdr_k not in student._direct_map:
                student._direct_map[sdr_k] = {}
            
            dm = student._direct_map[sdr_k]
            actual = input_ids[i+1]
            
            # 💡 チャットの記憶は優先して引き出せるよう、重みを強烈(500.0)に設定する
            dm[actual] = dm.get(actual, 0.0) + 500.0
            
            top_probs, top_indices = torch.topk(probs[i], 5)
            for rank in range(5):
                t_idx = top_indices[rank].item()
                if t_idx != actual:
                    dm[t_idx] = dm.get(t_idx, 0.0) + 50.0 * top_probs[rank].item()
                    
            for tok_id in list(dm.keys()):
                if tok_id != actual:
                    dm[tok_id] *= 0.8  
                if dm[tok_id] > 1000.0:
                    dm[tok_id] = 1000.0

    print("Saving updated memory...")
    state = {
        "direct_map": {str(k): {str(tk): v for tk, v in tv.items()} for k, tv in student._direct_map.items()},
        "vocab_size": student.vocab_size
    }
    with open(model_path, "wb") as f:
        msgpack.pack(state, f)
        
    print("✨ 対話学習が完了しました！")

if __name__ == "__main__":
    train_chat_data()