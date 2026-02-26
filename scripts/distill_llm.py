{
    "//": "ディレクトリパス: scripts/distill_llm.py",
    "//": "ファイルの日本語タイトル: LLM蒸留スクリプト (大容量コーパス・中断再開対応版)",
    "//": "ファイルの目的や内容: 既存のMessagePackモデルを読み込み、新しいコーパスの知識を追加する。さらに、長時間の学習を考慮して定期的な保存（チェックポイント）と、中断した箇所からの再開機能（Ctrl+Cキャッチ）を実装。"
}

import torch
import msgpack
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM
import tqdm

class SNNLLMDistiller:
    def __init__(
        self, 
        teacher_model_name: str, 
        student_model: SpikingLLM,
        device: str = "cpu"
    ):
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

    def load_student(self, path: str):
        if os.path.exists(path):
            print(f"Loading existing SNN memory from {path}...")
            with open(path, "rb") as f:
                state = msgpack.unpack(f, raw=False)
            
            fixed_map = {}
            for str_sdr_k, next_tokens in state["direct_map"].items():
                sdr_k = eval(str_sdr_k)
                fixed_map[sdr_k] = {int(k): float(v) for k, v in next_tokens.items()}
            
            self.student._direct_map = fixed_map
            print(f"Successfully loaded {len(self.student._direct_map)} existing context patterns.")
        else:
            print(f"No existing memory found at {path}. Starting fresh.")

    def distill_single_text(self, text: str, max_length: int = 128, top_k: int = 5):
        # 1文単位の蒸留処理に分離
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        input_ids = inputs["input_ids"][0].tolist()

        if len(input_ids) < 2:
            return

        with torch.no_grad():
            outputs = self.teacher(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        context_tokens = []
        context_window = 8

        for i in range(len(input_ids) - 1):
            current_tok = input_ids[i]
            actual_next_token = input_ids[i + 1]

            context_tokens.append(current_tok)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)

            top_probs, top_indices = torch.topk(probs[i], top_k)
            
            sdr = self.student._encode_to_sdr(context_tokens)
            sdr_k = self.student._sdr_key(sdr)

            if sdr_k not in self.student._direct_map:
                self.student._direct_map[sdr_k] = {}
            
            dm = self.student._direct_map[sdr_k]

            dm[actual_next_token] = dm.get(actual_next_token, 0.0) + 100.0

            for rank in range(top_k):
                target_token = top_indices[rank].item()
                target_prob = top_probs[rank].item()
                
                if target_token != actual_next_token:
                    increment = 10.0 * target_prob
                    dm[target_token] = dm.get(target_token, 0.0) + increment

            for tok_id in list(dm.keys()):
                if tok_id != actual_next_token:
                    dm[tok_id] *= 0.8
                if dm[tok_id] > 200.0:
                    dm[tok_id] = 200.0

    def save_student(self, path: str):
        state = {
            "direct_map": {str(k): {str(tk): v for tk, v in tv.items()} for k, tv in self.student._direct_map.items()},
            "vocab_size": self.student.vocab_size
        }
        with open(path, "wb") as f:
            msgpack.pack(state, f)

def load_corpus(filepath: str) -> list[str]:
    texts = []
    if os.path.exists(filepath):
        print(f"Loading corpus from {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        print(f"Loaded {len(texts)} lines.")
    else:
        print(f"Warning: {filepath} not found.")
    return texts

def load_progress(progress_file: str) -> int:
    """進捗状況を保存したJSONから、最後に処理した行番号を取得する"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
                return data.get("last_processed_index", 0)
        except Exception:
            return 0
    return 0

def save_progress(progress_file: str, index: int):
    """進捗状況をJSONに保存する"""
    with open(progress_file, "w") as f:
        json.dump({"last_processed_index": index}, f)

if __name__ == "__main__":
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000) 
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    distiller = SNNLLMDistiller(
        teacher_model_name="google/gemma-2-2b",
        student_model=student,
        device=device
    )

    model_path = "distilled_sara_llm.msgpack"
    progress_file = "data/progress.json"
    corpus_file = "data/corpus.txt"

    distiller.load_student(model_path)
    dataset = load_corpus(corpus_file)
    
    if dataset:
        start_index = load_progress(progress_file)
        total_lines = len(dataset)
        
        if start_index >= total_lines:
            print("✅ このコーパスはすべて学習済みです。新しいテキストを追加するか、progress.json を削除してください。")
        else:
            print(f"🚀 学習を再開します: {start_index + 1} 行目から {total_lines} 行目まで")
            
            # 定期保存の間隔（何行ごとにセーブするか）
            save_interval = 100
            
            try:
                for i in tqdm.tqdm(range(start_index, total_lines), desc="Distilling", initial=start_index, total=total_lines):
                    distiller.distill_single_text(dataset[i])
                    
                    # 定期保存処理 (100行ごと)
                    if (i + 1) % save_interval == 0:
                        distiller.save_student(model_path)
                        save_progress(progress_file, i + 1)
                
                # ループが最後まで終わった時の最終保存
                distiller.save_student(model_path)
                save_progress(progress_file, total_lines)
                print(f"✅ すべての学習が完了し、モデルを保存しました。")
                
            except KeyboardInterrupt:
                # Ctrl+C で安全に中断する処理
                print("\n⚠️ ユーザーによって学習が中断されました。")
                print("現在の状態を保存しています...")
                distiller.save_student(model_path)
                save_progress(progress_file, i)
                print(f"✅ {i}行目までの進捗を保存しました。次回は続きから再開できます。")
    else:
        print("❌ コーパスが空のため、蒸留をスキップしました。")