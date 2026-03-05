{
    "//": "ディレクトリパス: scripts/eval/chat_snn_lm.py",
    "//": "ファイルの日本語タイトル: SNN言語モデル 推論・対話スクリプト (デバッグ対応版)",
    "//": "ファイルの目的や内容: トークナイザーを導入してサブワードレベルでの推論に対応。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel
from src.sara_engine.utils.tokenizer import SaraTokenizer

def chat_loop(model_dir: str, debug_mode: bool = False):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Inference (Subword-Level)")
    if debug_mode:
        print("[DEBUG MODE ENABLED] Model will output internal potentials.")
    print("=" * 60)
    
    if not os.path.exists(model_dir):
        print("Error: Pre-trained model not found.")
        return

    print("Waking up SARA... Loading synaptic weights...")
    model = SpikingTransformerModel.from_pretrained(model_dir)
    
    tokenizer = SaraTokenizer(vocab_size=model.config.vocab_size, model_path=os.path.join(model_dir, "sara_vocab.json"))
    
    print("SARA is ready! (Type 'quit' or 'exit' to stop)\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            input_tokens = tokenizer.encode(user_input)
            
            generated_tokens, debug_logs = model.generate(
                input_ids=input_tokens, 
                max_length=150,
                temperature=0.2,
                fire_threshold=0.8,
                debug=debug_mode
            )
            
            response_text = tokenizer.decode(generated_tokens)
            print(f"SARA: {response_text}\n")

            if debug_mode and debug_logs:
                last_log = debug_logs[-1]
                if last_log.get("stop_reason"):
                    print(f"  [DEBUG] Stopped because: {last_log['stop_reason']}")
                if last_log.get("top_k"):
                    candidates = [(tokenizer.id_to_token.get(tid, "?"), round(pot, 2)) for tid, pot in last_log["top_k"]]
                    print(f"  [DEBUG] Final candidates before stop: {candidates}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    SAVE_DIRECTORY = "models/snn_lm_pretrained"
    ENABLE_DEBUG = True 
    chat_loop(model_dir=SAVE_DIRECTORY, debug_mode=ENABLE_DEBUG)