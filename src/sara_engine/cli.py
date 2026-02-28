{
    "//": "ディレクトリパス: src/sara_engine/cli.py",
    "//": "ファイルの日本語タイトル: SARA CLI エントリポイント",
    "//": "ファイルの目的や内容: ターミナルから `sara-chat` や `sara-train` コマンドで推論・学習を実行する。推論パラメータを外部から調整可能な引数を追加。"
}

import argparse
import time
import os
import json
import torch
import msgpack
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sara_engine.inference import SaraInference
from sara_engine.models.spiking_llm import SpikingLLM

def chat():
    """Start interactive dialogue from terminal with SARA"""
    parser = argparse.ArgumentParser(description="SARA Hippocampus Chat Engine")
    parser.add_argument("--model", type=str, default="models/distilled_sara_llm.msgpack", help="Path to the model file")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=3, help="Top K sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Biological refractory period penalty")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generated tokens")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[Error] Model file not found: {args.model}")
        print("Please check the execution directory.")
        return

    print("Loading SARA Engine...")
    sara = SaraInference(model_path=args.model)
    
    print("Ready! Type 'quit' or 'exit' to stop.")
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError): 
            break
            
        if user_input.strip().lower() in ["quit", "exit"]:
            print("SARA: Goodbye! Let's talk again.")
            break
            
        if not user_input.strip(): 
            continue
        
        sara.reset_buffer()
        start_time = time.time()
        prompt = f"You: {user_input}\nSARA:"
        
        response = sara.generate(
            prompt, 
            max_length=args.max_length, 
            top_k=args.top_k, 
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            stop_conditions=["\n"]
        )
        
        elapsed_time = time.time() - start_time
        
        if not response:
            print("SARA: (I have no memory of this)")
        else:
            clean_response = response.replace('\n', '')
            print(f"SARA: {clean_response}  [⏱️ {elapsed_time:.3f}s]")

def train():
    """Train SARA's personality from JSONL data"""
    parser = argparse.ArgumentParser(description="SARA Dialogue Trainer")
    parser.add_argument("data", type=str, help="Path to JSONL training data (e.g. data/chat_data.jsonl)")
    parser.add_argument("--model", type=str, default="models/distilled_sara_llm.msgpack", help="Path to save the model")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[Error] Training data not found: {args.data}")
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
    
    if os.path.exists(args.model):
        print(f"Opening SNN memory file: {args.model}...")
        with open(args.model, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        
        raw_map = state.get("direct_map", {})
        fixed_map = {eval(k): {int(tk): float(tv) for tk, tv in v.items()} for k, v in raw_map.items()}
        student._direct_map = fixed_map
        print(f"Loaded {len(fixed_map)} patterns.")
    else:
        print("[Warning] No existing memory found. Creating new brain...")
        student._direct_map = {}

    chat_lines = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chat_lines.append(json.loads(line))
                
    print(f"Start learning {len(chat_lines)} dialogue items...")
    
    for item in tqdm.tqdm(chat_lines, desc="Chat Training"):
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
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    state = {
        "direct_map": {str(k): {str(tk): v for tk, v in tv.items()} for k, tv in student._direct_map.items()},
        "vocab_size": student.vocab_size
    }
    with open(args.model, "wb") as f:
        msgpack.pack(state, f)
        
    print("Dialogue training completed successfully!")