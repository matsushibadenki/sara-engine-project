# ディレクトリパス: scripts/eval/chat_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 推論・対話スクリプト (デバッグ対応版)
# ファイルの目的や内容: ターミナル上のゴミ文字が残るUIバグを完全に排除し、原因究明のためのデバッグモードを追加。

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel

def decode_tokens(tokens: list[int]) -> str:
    chars = []
    for t in tokens:
        if 0 <= t <= 0x10FFFF:
            chars.append(chr(t))
    return "".join(chars)

def chat_loop(model_dir: str, debug_mode: bool = False):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Inference (Character-Level)")
    if debug_mode:
        print("[DEBUG MODE ENABLED] Model will output internal potentials.")
    print("=" * 60)
    
    if not os.path.exists(model_dir):
        print("Error: Pre-trained model not found.")
        return

    print("Waking up SARA... Loading synaptic weights...")
    model = SpikingTransformerModel.from_pretrained(model_dir)
    print("SARA is ready! (Type 'quit' or 'exit' to stop)\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            input_tokens = [ord(c) for c in user_input]
            
            # UIバグの元凶だった「thinking...」のアニメーションを削除し、純粋に待機する
            generated_tokens, debug_logs = model.generate(
                input_ids=input_tokens, 
                max_length=150,
                temperature=0.2,
                fire_threshold=0.8, # 少し閾値を下げて生成を繋がりやすくする
                debug=debug_mode
            )
            
            response_text = decode_tokens(generated_tokens)
            print(f"SARA: {user_input}{response_text}\n")

            # デバッグモードが有効な場合、なぜ生成が止まったのかを可視化する
            if debug_mode and debug_logs:
                last_log = debug_logs[-1]
                if last_log.get("stop_reason"):
                    print(f"  [DEBUG] Stopped because: {last_log['stop_reason']}")
                if last_log.get("top_k"):
                    candidates = [(chr(tid) if 0<=tid<=0x10FFFF else "?", round(pot, 2)) for tid, pot in last_log["top_k"]]
                    print(f"  [DEBUG] Final candidates before stop: {candidates}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    SAVE_DIRECTORY = "models/snn_lm_pretrained"
    # デバッグ情報を表示したい場合はここを True に変更してください
    ENABLE_DEBUG = True 
    chat_loop(model_dir=SAVE_DIRECTORY, debug_mode=ENABLE_DEBUG)