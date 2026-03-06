# {
#     "//": "ディレクトリパス: scripts/eval/chat_snn_lm.py",
#     "//": "ファイルの日本語タイトル: SNN言語モデル 推論・対話スクリプト (パラメータ調整版)",
#     "//": "ファイルの目的や内容: 未知の入力に対する沈黙を防ぐため、発火閾値を下げ、出力を安定させるために温度パラメータを調整。"
# }

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
    
    print("SARA is ready! (Type 'quit' or 'exit' to stop)")
    print("💡ヒント: 学習データ（AIやネットワークに関する語彙）に含まれる言葉を入力すると反応しやすくなります。\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            input_tokens = tokenizer.encode(user_input)
            
            if debug_mode:
                # ユーザーの入力がどのようにトークン化されたかを確認
                token_strs = [tokenizer.id_to_token.get(t, "?") for t in input_tokens]
                print(f"  [DEBUG] Input tokens: {token_strs}")

            # 無発話時は閾値を緩めて再試行し、沈黙を防ぐ
            decode_source_tokens = []
            debug_logs = []
            for fire_threshold, temperature in [(0.4, 0.1), (0.3, 0.2), (0.2, 0.35)]:
                generated_tokens, debug_logs = model.generate(
                    prompt=input_tokens,
                    max_length=150,
                    temperature=temperature,
                    fire_threshold=fire_threshold,
                    debug=debug_mode
                )
                new_generated_tokens = generated_tokens[len(input_tokens):]
                if new_generated_tokens:
                    decode_source_tokens = new_generated_tokens
                    break

            response_text = tokenizer.decode(decode_source_tokens)
            
            # 生成テキストが空の場合は「...」を表示
            if not response_text.strip():
                response_text = "..."
                
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
