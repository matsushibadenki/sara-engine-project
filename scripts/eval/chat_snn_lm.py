from sara_engine.utils.tokenizer import SaraTokenizer
from sara_engine.utils.chat import ChatSessionHelper
from sara_engine.models.snn_transformer import SpikingTransformerModel
from typing import List, Tuple
import re
import argparse
import sys
import os
# ディレクトリパス: scripts/eval/chat_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 推論・対話スクリプト (パラメータ調整版)
# ファイルの目的や内容: 英語ノイズや無意味な出力を防ぐため、スコアリングのペナルティ強化およびサニタイズ処理を追加。


sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'src')))


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
    quote_count = stripped.count(
        "「") + stripped.count("」") + stripped.count("『") + stripped.count("』")
    citation_like = len(re.findall(r"(第\d+巻|第\d+号|\d{4}年|\d+-\d+頁)", stripped))

    repeated_phrase_penalty = 0.0
    for span in range(8, min(24, max(9, len(stripped) // 2))):
        seen = {}
        for i in range(0, max(0, len(stripped) - span + 1)):
            frag = stripped[i:i + span]
            if len(frag.strip()) < max(4, span // 2):
                continue
            seen[frag] = seen.get(frag, 0) + 1
        repeated_phrase_penalty += sum((count - 1) *
                                       2.5 for count in seen.values() if count > 1)

    score = float(jp_count) * 1.2
    # 英字に対するペナルティを強化し、英語のノイズ出力を抑止
    score -= float(ascii_count) * 1.5
    score -= float(digit_count) * 0.2
    score -= float(noise_count) * 1.5
    score -= float(quote_count) * 0.2
    score -= float(citation_like) * 2.0
    score -= repeated_phrase_penalty
    score -= max(0, line_breaks - 2) * 3.0

    if stripped[-1] in "。！？":
        score += 3.0
    if len(stripped) > 220:
        score -= 2.0
    return score


def _clean_response(text: str, max_chars: int = 200) -> str:
    if not text:
        return text
    normalized = text.replace("<eos>", "").replace(
        "<sos>", "").replace("<pad>", "").replace("<unk>", "")
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # 英語のノイズ（ハイフンとアルファベットの連続）を削除
    normalized = re.sub(r"[-a-zA-Z\s]{15,}.*$", "", normalized)
    normalized = re.sub(
        r"(『[^』]{0,40}|第\d+巻.*$|第\d+号.*$|\d{4}年.*$)", "", normalized)
    normalized = normalized.replace("、。", "。")

    paired_marks = [("「", "」"), ("『", "』"), ("(", ")"), ("（", "）")]
    for opener, closer in paired_marks:
        if normalized.count(opener) > normalized.count(closer):
            cut = normalized.rfind(opener)
            if cut > 8:
                normalized = normalized[:cut].rstrip()

    if len(normalized) > max_chars:
        normalized = normalized[:max_chars].rstrip()
        last_punc = max(normalized.rfind(
            "。"), normalized.rfind("！"), normalized.rfind("？"))
        if last_punc > 30:
            normalized = normalized[:last_punc + 1]
    if normalized and normalized[-1] in "、・「『（(":
        normalized = normalized[:-1].rstrip()
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

    tokenizer = SaraTokenizer(vocab_size=model.config.vocab_size,
                              model_path=os.path.join(model_dir, "sara_vocab.json"))
    chat_helper = ChatSessionHelper(max_turns=4)

    print("SARA is ready! (Type 'quit' or 'exit' to stop)")
    print("💡ヒント: 学習データ（AIやネットワークに関する語彙）に含まれる言葉を入力すると反応しやすくなります。\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            prompt_text = chat_helper.build_prompt_text(user_input)
            input_tokens = tokenizer.encode(prompt_text)

            if debug_mode:
                # ユーザーの入力がどのようにトークン化されたかを確認
                token_strs = [tokenizer.id_to_token.get(
                    t, "?") for t in input_tokens]
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

                response_text = _decode_new_tokens(
                    tokenizer, generated_tokens, len(input_tokens))
                cleaned_text = _clean_response(response_text)
                score = _score_response(
                    cleaned_text) + chat_helper.rerank_score(user_input, cleaned_text)
                if score > best_score:
                    best_score = score
                    best_response = cleaned_text
                    best_logs = debug_logs
                if cleaned_text and cleaned_text.endswith(("。", "！", "？")) and score >= 10:
                    break

            if not best_response.strip() or best_score < 2.0:
                best_response = chat_helper.fallback_response(user_input)

            print(f"SARA: {best_response}\n")
            chat_helper.add_turn("user", user_input)
            chat_helper.add_turn("assistant", best_response)

            if debug_mode and best_logs:
                last_log = best_logs[-1]
                if last_log.get("stop_reason"):
                    print(
                        f"  [DEBUG] Stopped because: {last_log['stop_reason']}")
                if last_log.get("top_k"):
                    candidates = [(tokenizer.id_to_token.get(tid, "?"), round(
                        pot, 2)) for tid, pot in last_log["top_k"]]
                    print(
                        f"  [DEBUG] Final candidates before stop: {candidates}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive chat for SNN language model.")
    parser.add_argument("--model-dir", default="models/snn_lm_pretrained",
                        help="Directory of pretrained model.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output.")
    parser.add_argument("--max-length", type=int, default=64,
                        help="Base maximum length for generated tokens.")
    args = parser.parse_args()

    chat_loop(
        model_dir=args.model_dir,
        debug_mode=args.debug,
        max_length=max(16, args.max_length),
    )
