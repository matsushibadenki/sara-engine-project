{
    "//": "ディレクトリパス: scripts/distill_llm.py",
    "//": "ファイルの日本語タイトル: LLM蒸留スクリプト (メモリ安全・8192ニューロン版)",
    "//": "ファイルの目的や内容: メモリ超過による強制終了を防ぐためニューロン数を8192に戻し、忘却ロジックを用いて安全に蒸留する。"
}

import torch
import os
import json
import tqdm
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
project_root = os.path.abspath(os.path.join(scripts_dir, ".."))
src_dir = os.path.join(project_root, "src")
for path in (scripts_dir, src_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.utils.project_paths import interim_data_path, model_path
from utils.manage_db import SaraCorpusDB

class SNNLLMDistiller:
    def __init__(self, teacher_model_name, student_model, device="cpu"):
        print(f"Loading teacher model: {teacher_model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, 
            torch_dtype=torch.float32, 
            device_map=device
        )
        self.teacher.eval()
        self.student = student_model
        self.device = device

    def load_student(self, path):
        if os.path.exists(path):
            print(f"Opening SNN memory file: {path}...")
            loaded_count = self.student.load_memory(path)
            print(f"Restoring {loaded_count} context patterns...")
            print(f"✅ Successfully loaded memory.")
        else:
            print(f"No existing memory found at {path}. Starting fresh.")

    def save_student(self, path):
        self.student.save_memory(path)

    def distill_single_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        input_ids = inputs["input_ids"][0].tolist()
        if len(input_ids) < 2: return

        with torch.no_grad():
            outputs = self.teacher(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        context_tokens = []
        for i in range(len(input_ids) - 1):
            context_tokens.append(input_ids[i])
            if len(context_tokens) > 8: context_tokens.pop(0)
            
            sdr_k = self.student._sdr_key(self.student._encode_to_sdr(context_tokens))
            if sdr_k not in self.student._direct_map:
                self.student._direct_map[sdr_k] = {}
            
            dm = self.student._direct_map[sdr_k]
            actual = input_ids[i+1]
            
            dm[actual] = dm.get(actual, 0.0) + 100.0
            
            top_probs, top_indices = torch.topk(probs[i], 5)
            for rank in range(5):
                t_idx = top_indices[rank].item()
                if t_idx != actual:
                    dm[t_idx] = dm.get(t_idx, 0.0) + 10.0 * top_probs[rank].item()
                    
            for tok_id in list(dm.keys()):
                if tok_id != actual:
                    dm[tok_id] *= 0.8  
                if dm[tok_id] > 200.0:
                    dm[tok_id] = 200.0

if __name__ == "__main__":
    student_model_path = model_path("distilled_sara_llm.msgpack")
    progress_file = interim_data_path("distill_progress.json")
    
    print("Initializing SNN Student Model (8192 neurons)...")
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    distiller = SNNLLMDistiller("google/gemma-2-2b", student, device)
    distiller.load_student(student_model_path)

    db = SaraCorpusDB()
    last_id = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            last_id = json.load(f).get("last_id", 0)

    print(f"🚀 Distilling from DB (Starting ID: {last_id})")
    
    try:
        cur = db.conn.execute("SELECT id, content FROM corpus WHERE id > ? ORDER BY id", (last_id,))
        rows = cur.fetchall()
        
        if not rows:
            print("✅ No new data to distill.")
        else:
            for i, row in enumerate(tqdm.tqdm(rows, desc="Overall Progress")):
                distiller.distill_single_text(row[1])
                
                if (i + 1) % 50 == 0:
                    distiller.save_student(student_model_path)
                    with open(progress_file, "w") as f:
                        json.dump({"last_id": row[0]}, f)
            
            distiller.save_student(student_model_path)
            with open(progress_file, "w") as f:
                json.dump({"last_id": rows[-1][0]}, f)
            print("✨ Distillation completed successfully.")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted. Saving current progress...")
        distiller.save_student(student_model_path)
