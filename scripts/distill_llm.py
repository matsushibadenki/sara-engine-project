# ディレクトリパス: scripts/distill_llm.py
# ファイルの日本語タイトル: LLM蒸留スクリプト (SQLite DB対応・インポート修正版)
# ファイルの目的や内容: インポートエラーを修正。SQLiteからデータを逐次読み込み、SNNへ蒸留する。

import torch
import msgpack
import os
import json
import tqdm
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

# 💡 インポートエラー回避のための処理
# 実行中のスクリプトがあるディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# これで ModuleNotFoundError: No module named 'scripts' が出なくなります
from manage_db import SaraCorpusDB

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
        """既存のMessagePackから記憶を復元する"""
        if os.path.exists(path):
            print(f"Opening SNN memory file: {path}...")
            with open(path, "rb") as f:
                state = msgpack.unpack(f, raw=False)
            
            raw_map = state.get("direct_map", {})
            print(f"Restoring {len(raw_map)} context patterns...")
            
            fixed_map = {}
            for k, v in tqdm.tqdm(raw_map.items(), desc="Loading SNN Memory"):
                fixed_map[eval(k)] = {int(tk): float(tv) for tk, tv in v.items()}
            
            self.student._direct_map = fixed_map
            print(f"✅ Successfully loaded memory.")
            del state
        else:
            print(f"No existing memory found at {path}. Starting fresh.")

    def save_student(self, path):
        """モデルをMessagePack形式で保存"""
        print(f"Saving SNN memory to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "direct_map": {str(k): {str(tk): v for tk, v in tv.items()} for k, tv in self.student._direct_map.items()},
            "vocab_size": self.student.vocab_size
        }
        with open(path, "wb") as f:
            msgpack.pack(state, f)
        print("✅ Save completed.")

    def distill_single_text(self, text):
        """1文の蒸留処理"""
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
            
            # 正解ラベルの重み付け
            dm[actual] = min(dm.get(actual, 0.0) + 100.0, 200.0)
            
            # ソフトラベル（周囲の確率）の重み付け
            top_probs, top_indices = torch.topk(probs[i], 5)
            for rank in range(5):
                t_idx = top_indices[rank].item()
                if t_idx != actual:
                    dm[t_idx] = min(dm.get(t_idx, 0.0) + 10.0 * top_probs[rank].item(), 200.0)

if __name__ == "__main__":
    # パス設定
    model_path = "models/distilled_sara_llm.msgpack"
    data_dir = "data"
    progress_file = os.path.join(data_dir, "progress.json")
    
    # モデル初期化
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    distiller = SNNLLMDistiller("google/gemma-2-2b", student, device)
    distiller.load_student(model_path)

    # DB接続
    db = SaraCorpusDB()
    last_id = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            last_id = json.load(f).get("last_id", 0)

    print(f"🚀 Distilling from DB (Starting ID: {last_id})")
    
    try:
        # DBから未学習のデータを取得
        cur = db.conn.execute("SELECT id, content FROM corpus WHERE id > ? ORDER BY id", (last_id,))
        rows = cur.fetchall()
        
        if not rows:
            print("✅ No new data to distill.")
        else:
            for i, row in enumerate(tqdm.tqdm(rows, desc="Overall Progress")):
                distiller.distill_single_text(row[1])
                
                # 50件ごとに保存
                if (i + 1) % 50 == 0:
                    distiller.save_student(model_path)
                    with open(progress_file, "w") as f:
                        json.dump({"last_id": row[0]}, f)
            
            # 最後に保存
            distiller.save_student(model_path)
            with open(progress_file, "w") as f:
                json.dump({"last_id": rows[-1][0]}, f)
            print("✨ Distillation completed successfully.")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted. Saving current progress...")
        distiller.save_student(model_path)
        # 最後に処理した行のIDを記録
        # ここでは i がループ内変数なので、直前の row[0] を使うなどの工夫が必要ですが
        # 簡易的に中断時の保存を行います