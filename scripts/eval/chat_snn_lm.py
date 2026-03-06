# {
#     "//": "ディレクトリパス: scripts/eval/chat_snn_lm.py",
#     "//": "ファイルの日本語タイトル: SNN言語モデル 推論・対話スクリプト (パラメータ調整版)",
#     "//": "ファイルの目的や内容: 未知の入力に対する沈黙を防ぐため、発火閾値を下げ、出力を安定させるために温度パラメータを調整。"
# }

import os
import sys
import argparse
import re
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel
from src.sara_engine.utils.tokenizer import SaraTokenizer

def _score_response(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return -1e9

    jp_count = sum(1 for ch in stripped if (
        '\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff'
    ))
    ascii_count = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
    digit_count = sum(1 for ch in stripped if ch.isdigit())
    noise_count = len(re.findall(r"[{}[\]<>|\\/@#$%^*_+=~`]", stripped))
    line_breaks = stripped.count("\n")

    score = float(jp_count) * 1.2
    score -= float(ascii_count) * 0.5
    score -= float(digit_count) * 0.2
    score -= float(noise_count) * 1.5
    score -= max(0, line_breaks - 2) * 3.0
    if stripped[-1] in "。！？":
        score += 3.0
    if len(stripped) > 220:
        score -= 2.0
    return score


def _clean_response(text: str, max_chars: int = 200) -> str:
    if not text:
        return text
    normalized = text.replace("<eos>", "").replace("<sos>", "").replace("<pad>", "").replace("<unk>", "")
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if len(normalized) > max_chars:
        normalized = normalized[:max_chars].rstrip()
        last_punc = max(normalized.rfind("。"), normalized.rfind("！"), normalized.rfind("？"))
        if last_punc > 30:
            normalized = normalized[:last_punc + 1]
    return normalized.strip()


def _decode_new_tokens(tokenizer: SaraTokenizer, generated_tokens: List[int], input_len: int) -> str:
    new_tokens = generated_tokens[input_len:]
    if not new_tokens:
        return ""
    return tokenizer.decode(new_tokens)


def chat_loop(
    model_dir: str,
    debug_mode: bool = False,
    max_length: int = 64,
):
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

            retry_settings: List[Tuple[float, float, int]] = [
                (0.55, 0.06, max_length),
                (0.45, 0.12, int(max_length * 1.15)),
                (0.35, 0.20, int(max_length * 1.30)),
            ]

            best_response = ""
            best_score = -1e9
            best_logs = []

            for fire_threshold, temperature, retry_max_len in retry_settings:
                generated_tokens, debug_logs = model.generate(
                    prompt=input_tokens,
                    max_length=retry_max_len,
                    temperature=temperature,
                    fire_threshold=fire_threshold,
                    debug=debug_mode
                )

                response_text = _decode_new_tokens(tokenizer, generated_tokens, len(input_tokens))
                cleaned_text = _clean_response(response_text)
                score = _score_response(cleaned_text)
                if score > best_score:
                    best_score = score
                    best_response = cleaned_text
                    best_logs = debug_logs
                if cleaned_text and cleaned_text.endswith(("。", "！", "？")) and score >= 10:
                    break
            
            # 生成テキストが空の場合は「...」を表示
            if not best_response.strip():
                best_response = "..."
                
            print(f"SARA: {best_response}\n")

            if debug_mode and best_logs:
                last_log = best_logs[-1]
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
    parser = argparse.ArgumentParser(description="Interactive chat for SNN language model.")
    parser.add_argument("--model-dir", default="models/snn_lm_pretrained", help="Directory of pretrained model.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    parser.add_argument("--max-length", type=int, default=64, help="Base maximum length for generated tokens.")
    args = parser.parse_args()

    chat_loop(
        model_dir=args.model_dir,
        debug_mode=args.debug,
        max_length=max(16, args.max_length),
    )
