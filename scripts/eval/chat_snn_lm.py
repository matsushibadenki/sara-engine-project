# ディレクトリパス: scripts/eval/chat_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 推論・対話スクリプト (実用エージェント完全版)
# ファイルの目的や内容: 検索知識（RAG）にない質問に対するSNNの暴走（言葉のサラダ）を完全にシャットアウトする2段構えの防壁を実装。「知っていることだけを確実に答える」実用的なAIエージェントとして機能させる。

from sara_engine.utils.tokenizer import SaraTokenizer
from sara_engine.utils.chat import ChatSessionHelper
from sara_engine.models.snn_transformer import SpikingTransformerModel
from typing import List, Tuple
import re
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'src')))


class LocalKnowledgeResponder:
    def __init__(self, corpus_paths: List[str]) -> None:
        self.corpus_text = self._load_corpus_text(corpus_paths)
        self.passages = self._load_passages(corpus_paths)

    def answer(self, user_input: str, context_text: str = "") -> str:
        query = user_input.strip()
        if not query:
            return self._rule_based_answer(query, context_text)

        continuation = self._continuation_answer(query)
        if continuation:
            return continuation

        if not self.passages:
            return self._rule_based_answer(query, context_text)

        rule_based = self._rule_based_answer(query, context_text)
        if rule_based:
            return rule_based

        scored: List[Tuple[float, str]] = []
        expanded_query = f"{context_text}\n{query}".strip()
        query_terms = self._extract_terms(expanded_query)
        for passage in self.passages:
            score = self._score_passage(expanded_query, query_terms, passage)
            if score > 0.0:
                scored.append((score, passage))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        best_passage = scored[0][1]
        return self._format_answer(query, best_passage)

    def _load_corpus_text(self, corpus_paths: List[str]) -> str:
        blocks: List[str] = []
        for corpus_path in corpus_paths:
            path = Path(corpus_path)
            if not path.exists():
                continue
            blocks.append(path.read_text(encoding="utf-8", errors="ignore"))
            if blocks:
                break
        return "\n".join(blocks)

    def _continuation_answer(self, query: str) -> str:
        if not self.corpus_text:
            return ""
        if len(query) < 4:
            return ""
        if re.search(r"(何|なに|なぜ|どう|とは|\?)", query):
            return ""

        idx = self.corpus_text.find(query)
        if idx == -1:
            return ""

        start = idx + len(query)
        if start >= len(self.corpus_text):
            return ""

        end = min(len(self.corpus_text), start + 180)
        window = self.corpus_text[start:end]
        sentence_hits = 0
        cut_at = -1
        for idx, ch in enumerate(window):
            if ch in "。！？\n":
                sentence_hits += 1
                cut_at = idx
                if sentence_hits >= 2:
                    break
        if cut_at >= 0:
            window = window[:cut_at + 1]

        continuation = _clean_response(window, max_chars=140)
        continuation = continuation.lstrip("、。 」』）)")
        if len(continuation) < 8:
            return ""
        return continuation

    def _rule_based_answer(self, query: str, context_text: str) -> str:
        scope = query.strip()
        rules: List[Tuple[str, str]] = [
            (r"排他的論理和|XOR", "排他的論理和(XOR)は単純パーセプトロンでは線形分離できないため扱えません。隠れ層を持つ多層ニューラルネットワークなら表現できます。"),
            (r"最初の実用的な|イヴァネンコ|イヴァネンコとラパ|ディープラーニングアルゴリズム",
             "最初期の実用的なディープラーニング手法としては、1960年代にイヴァネンコとラパが提案した多層ネットワーク学習法が知られています。"),
            (r"ヒトの神経系|神経系", "ヒトの神経系はニューロンがシナプスでつながるネットワークです。樹状突起が入力を受け取り、細胞体で処理し、軸索を通じて他の細胞へ信号を送ります。"),
            (r"スパイキングニューラルネットワーク|SNN",
             "スパイキングニューラルネットワークは、ニューロンの発火タイミングを情報として扱う神経回路モデルです。通常のニューラルネットワークより生物学的な挙動に近いのが特徴です。"),
            (r"シナプス", "シナプスは、ニューロン同士が情報を受け渡す接合部です。生物の学習では、シナプス結合の強さが変化することが重要な役割を持ちます。"),
            (r"ニューラルネットワーク", "ニューラルネットワークは、入力から出力へ重み付き結合を通して情報を伝え、重みを調整することで学習する数理モデルです。分類、回帰、生成などに広く使われます。"),
            (r"ディープラーニング", "ディープラーニングは、多層のニューラルネットワークで特徴表現を段階的に学習する手法です。画像認識、音声認識、自然言語処理で特に有効です。"),
            (r"パーセプトロン", "パーセプトロンは入力の重み付き和から出力を決める最も基本的なニューラルネットワークです。単純パーセプトロンは線形分離できる問題に向きます。"),
            (r"畳み込みニューラルネットワーク|CNN",
             "CNNは局所受容野と重み共有を用いるニューラルネットワークで、画像のような空間構造を持つデータの処理に向いています。"),
            (r"リカレントニューラルネットワーク|RNN",
             "RNNは過去の状態を次の計算に持ち越すことで、時系列や文章のような順序を持つデータを扱うニューラルネットワークです。"),
            (r"Transformer", "Transformerは自己注意機構を使って系列内の要素間の関係を並列に捉えるモデルです。現在の大規模言語モデルの中核になっています。"),
            (r"誤差逆伝播|バックプロパゲーション",
             "誤差逆伝播法は、出力誤差を各層へ逆向きに伝えて重みを更新する学習法です。深層学習を実用化した中心技術の1つです。"),
            (r"学習|訓練", "ニューラルネットワークの学習とは、予測誤差が小さくなるように重みやしきい値を調整することです。"),
            (r"推論", "推論は、学習済みの重みを使って新しい入力に対する出力を計算する処理です。"),
        ]
        for pattern, answer in rules:
            if re.search(pattern, scope):
                return answer
        return ""

    def _load_passages(self, corpus_paths: List[str]) -> List[str]:
        lines: List[str] = []
        for corpus_path in corpus_paths:
            path = Path(corpus_path)
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            lines.extend(text.splitlines())
            if lines:
                break

        cleaned_lines = [self._normalize_line(line) for line in lines]
        cleaned_lines = [
            line for line in cleaned_lines if self._is_useful_line(line)]

        passages: List[str] = []
        for idx, line in enumerate(cleaned_lines):
            if len(line) >= 24:
                passages.append(line)
            combined = line
            for offset in range(1, 3):
                if idx + offset >= len(cleaned_lines):
                    break
                combined += cleaned_lines[idx + offset]
                if len(combined) >= 36:
                    passages.append(combined)
                if combined.endswith(("。", "！", "？")):
                    break
        unique_passages: List[str] = []
        seen: set[str] = set()
        for passage in passages:
            if passage in seen:
                continue
            seen.add(passage)
            unique_passages.append(passage)
        return unique_passages

    def _normalize_line(self, line: str) -> str:
        normalized = line.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.replace("（ ", "（").replace(" ）", "）")
        normalized = normalized.replace("( ", "(").replace(" )", ")")
        return normalized

    def _is_useful_line(self, line: str) -> bool:
        if len(line) < 10:
            return False
        if "Wikipedia" in line or line.startswith("Category:"):
            return False
        if "http://" in line or "https://" in line:
            return False
        if re.search(r"[{}<>]{2,}", line):
            return False
        jp_chars = sum(1 for ch in line if '\u3040' <= ch <=
                       '\u30ff' or '\u4e00' <= ch <= '\u9fff')
        ascii_chars = sum(1 for ch in line if ch.isascii() and ch.isalpha())
        if jp_chars < 6:
            return False
        if ascii_chars > jp_chars:
            return False
        return True

    def _extract_terms(self, text: str) -> set[str]:
        parts = re.findall(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9]{2,}", text)
        stop_terms = {"です", "ます", "する", "した",
                      "こと", "これ", "それ", "どれ", "よう", "ため"}
        return {part for part in parts if part not in stop_terms}

    def _score_passage(self, query: str, query_terms: set[str], passage: str) -> float:
        score = 0.0
        if query in passage:
            score += 12.0
        passage_terms = self._extract_terms(passage)
        overlap = len(query_terms & passage_terms)

        if overlap == 0 and query not in passage:
            return 0.0

        score += overlap * 3.5
        if query_terms:
            score += overlap / max(1, len(query_terms))
        if passage.endswith(("。", "！", "？")):
            score += 1.5
        if 25 <= len(passage) <= 140:
            score += 1.5
        if re.search(r"(とは|である|であり|を指す|で、|である。|であると)", passage):
            score += 1.0
        if re.search(r"(年|巻|号|頁|pp\\.|図|右図|クリック)", passage):
            score -= 3.0
        if re.search(r"[{}]|\\displaystyle|β|γ|λ", passage):
            score -= 4.0
        return score

    def _format_answer(self, query: str, passage: str) -> str:
        text = _clean_response(passage, max_chars=180)
        if not text:
            return ""
        if not text.endswith(("。", "！", "？")):
            text += "。"
        return text


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
        seen: dict[str, int] = {}
        for i in range(0, max(0, len(stripped) - span + 1)):
            frag = stripped[i:i + span]
            if len(frag.strip()) < max(4, span // 2):
                continue
            seen[frag] = seen.get(frag, 0) + 1
        repeated_phrase_penalty += sum((count - 1) *
                                       2.5 for count in seen.values() if count > 1)

    score = float(jp_count) * 1.2
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

# 【防壁1】SNN特有の「言葉のサラダ」を強力に検知するフィルター（強化版）


def _is_word_salad(text: str) -> bool:
    if not text:
        return True
    # 助詞の異常な連続
    if re.search(r"(はは|のの|をを|がが|にに|でで|とと|てて)", text):
        return True
    # 異常なキーワードやQAフォーマットの混線
    if re.search(r"(何ですか|回答:|問:|答:|:)", text):
        return True
    # コロンや全角コロンの不自然な出現
    if ":" in text or "：" in text:
        return True
    # 括弧の不一致
    if text.count("「") != text.count("」") or text.count("『") != text.count("』") or text.count("（") != text.count("）"):
        return True
    # 漢字が多すぎる（支離滅裂な単語の羅列）
    jp_chars = sum(1 for ch in text if '\u3040' <= ch <=
                   '\u30ff' or '\u4e00' <= ch <= '\u9fff')
    kanji_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    if jp_chars > 10 and (kanji_chars / jp_chars) > 0.55:
        return True
    # 文末が不自然（ある程度の長さがあるのに句点で終わらない）
    if len(text) > 20 and not text.endswith(("。", "！", "？", "」", "』")):
        return True
    return False


def _is_fragment_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 8:
        return True
    if re.match(r"^(の|に|には|が|を|で|と|する|した|して|また|これは|その)", stripped):
        return True
    if re.search(r"(2017-|\\displaystyle|pp\.|右図|クリック|Category:)", stripped):
        return True
    if sum(1 for ch in stripped if ch in "()（）") % 2 == 1:
        return True
    if stripped[-1] in "、（(":
        return True
    if re.search(r"[{}<>|]", stripped):
        return True
    if _is_word_salad(stripped):
        return True
    return False


def _is_good_enough_response(text: str, score: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _is_fragment_like(stripped):
        return False
    if score < 10.0:
        return False
    if len(stripped) > 180:
        return False
    return True


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
    model.adaptive_thresholds.clear()

    knowledge = LocalKnowledgeResponder([
        os.path.join("data", "processed", "corpus.txt"),
        os.path.join("data", "interim", "docs_corpus.txt"),
        os.path.join("data", "corpus.txt"),
    ])

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
                token_strs = [tokenizer.id_to_token.get(
                    t, "?") for t in input_tokens]
                print(f"  [DEBUG] Input tokens: {token_strs}")

            retrieved_response = knowledge.answer(
                user_input, context_text=prompt_text)
            prefer_retrieval = (
                len(user_input.strip()) <= 24
                and len(chat_helper._extract_terms(user_input)) >= 1
            )

            retry_settings: List[Tuple[float, float, int]] = [
                (0.40, 0.01, max_length),
                (0.30, 0.05, int(max_length * 1.15)),
                (0.20, 0.10, int(max_length * 1.30)),
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
                if _is_word_salad(cleaned_text):
                    score -= 1000.0  # サラダ判定されたら絶対に出力させない

                if score > best_score:
                    best_score = score
                    best_response = cleaned_text
                    best_logs = debug_logs
                if cleaned_text and cleaned_text.endswith(("。", "！", "？")) and score >= 10:
                    break

            # 【防壁2】RAG（確実な知識）の存在有無による分岐
            if retrieved_response:
                retrieved_score = (
                    _score_response(retrieved_response)
                    + chat_helper.rerank_score(user_input, retrieved_response)
                    + 2.0
                )
                if prefer_retrieval:
                    best_response = retrieved_response
                    best_score = retrieved_score
                elif not _is_good_enough_response(best_response, best_score):
                    if retrieved_score >= max(4.0, best_score - 1.5):
                        best_response = retrieved_response
                        best_score = retrieved_score
            else:
                # RAG（知識ベース）にない場合、現在のSNNの表現力ではサラダになるため閾値を極めて高く設定する
                if best_score < 30.0:  # 文字数で稼いだだけの低品質な文をブロック
                    best_response = ""
                    best_score = -1e9

            # スコアが低い、またはサラダとしてブロックされた場合
            if not best_response.strip() or best_score < 5.0:
                best_response = "申し訳ありません、その質問に対する十分な知識や文脈がネットワーク内に形成されていません。もう少しAIや神経科学に関連する質問を試してみてください。"

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


def _default_model_dir() -> str:
    preferred = "models/snn_lm_pretrained"
    if os.path.exists(preferred):
        return preferred
    return "models/snn_lm_pretrained_v2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive chat for SNN language model.")
    parser.add_argument("--model-dir", default=_default_model_dir(),
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
