# examples/demo_spiking_causal_lm_chat.py
# v4: register_vocab を使って supervised_qa_train の句読点除外を有効化

from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.models.spiking_causal_lm import SpikingCausalLM
import json
import time
import os
import random
import math
from typing import List, Dict, Tuple

_FILE_INFO = {
    "path":  "examples/demo_spiking_causal_lm_chat.py",
    "title": "スパイキング因果LLMチャットデモ v4",
}

DATA_PATH      = "data/chat_data.jsonl"
MAX_SAMPLES    = 50
VOCAB_SIZE     = 500
EMBED_DIM      = 256
HIDDEN_DIM     = 512
SPARSITY       = 0.10
EPOCHS         = 30
LR_NORMAL      = 0.8
LR_QA          = 2.0
MAX_GEN_TOKENS = 25
TEMPERATURE    = 0.2
DEBUG          = True
SEP            = " Assistant: "


def load_data(data_path: str, max_samples: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    qa_pairs: List[Tuple[str, str]] = []
    if not os.path.exists(data_path):
        print(f"[WARN] {data_path} not found. Using built-in samples.")
        qa_pairs = [
            ("こんにちは",    "こんにちは！SARAです。何かお手伝いしましょうか？"),
            ("調子はどう？",  "絶好調です！M4チップの推論速度は素晴らしいですね。"),
            ("あなたは誰？",  "私はSARAです。ローカル環境で動くSNNのAIです。"),
            ("ありがとう",    "どういたしまして！また何かあれば聞いてください。"),
            ("何ができるの？","テキスト生成や質問応答など、様々なタスクに対応しています。"),
        ]
    else:
        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    u = data["user"]
                    a = data.get("assistant", "") or data.get("sara", "")
                    if u and a:
                        qa_pairs.append((u, a))
                except (json.JSONDecodeError, KeyError):
                    continue

    corpus = [f"User: {q}{SEP}{a}" for q, a in qa_pairs]
    return corpus, qa_pairs


def patch_sparsity(model: SpikingCausalLM, sparsity: float) -> None:
    def _patched(token_id: int, sparsity: float = sparsity) -> List[int]:
        if token_id not in model.token_to_sdr:
            random.seed(token_id)
            num_spikes = max(1, int(model.embed_dim * sparsity))
            model.token_to_sdr[token_id] = random.sample(
                range(model.embed_dim), num_spikes)
            random.seed()
        return model.token_to_sdr[token_id]
    model._get_sdr_for_token = _patched  # type: ignore


def build_id_to_token(tokenizer: SpikeTokenizer) -> Dict[int, str]:
    """
    SpikeTokenizer から id→token テキストの逆引きテーブルを構築する。

    優先順位:
      1. get_vocab() が使えて非空エントリが多ければそれを使う
      2. vocab_size 分ループして decode() で引く
      3. それでも空なら tokenizer の内部属性 (decoder 等) を直接参照する
    """
    id_to_token: Dict[int, str] = {}

    # 方法1: get_vocab() → {token_str: id} の逆引き
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        candidate: Dict[int, str] = {v: k for k, v in vocab.items()}
        non_empty_c = sum(1 for v in candidate.values() if v.strip())
        if non_empty_c > len(candidate) * 0.3:
            print(f"  [build_id_to_token] get_vocab: {non_empty_c}/{len(candidate)} non-empty")
            return candidate

    # 方法2: decode([i]) で個別に引く
    for i in range(tokenizer.vocab_size):
        try:
            text = tokenizer.decode([i])
            id_to_token[i] = text if text is not None else ""
        except Exception:
            id_to_token[i] = ""

    non_empty2 = sum(1 for v in id_to_token.values() if v.strip())

    # 方法2が不十分なら方法3: 内部属性を直接参照
    if non_empty2 < tokenizer.vocab_size * 0.3:
        for attr in ("id_to_token", "_id_to_token", "decoder", "_decoder"):
            if hasattr(tokenizer, attr):
                raw = getattr(tokenizer, attr)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(k, int) and isinstance(v, str):
                            id_to_token[k] = v
                    non_empty3 = sum(1 for v in id_to_token.values() if v.strip())
                    if non_empty3 > tokenizer.vocab_size * 0.3:
                        print(f"  [build_id_to_token] attr '{attr}': {non_empty3} non-empty")
                        break

    non_empty_f = sum(1 for v in id_to_token.values() if v.strip())
    print(f"  [build_id_to_token] final: {non_empty_f}/{tokenizer.vocab_size} non-empty")

    # それでも非空が少ない場合: encode→decode往復でサンプル確認
    if non_empty_f < 10:
        print("  [build_id_to_token] WARNING: very few non-empty entries!")
        print("  Trying encode→id mapping for known texts...")
        # 回答テキストから直接マッピングを作る
        known_texts = [
            "こんにちは", "SARAです", "絶好調です", "私はSARAです",
            "どういたしまして", "様々なタスクに対応しています",
            "ローカル環境で動くSNNのAIです", "また何かあれば聞いてください",
            "M4チップの推論速度は素晴らしいですね",
        ]
        for text in known_texts:
            ids = tokenizer.encode(text)
            if len(ids) == 1:
                id_to_token[ids[0]] = text
        non_empty_f2 = sum(1 for v in id_to_token.values() if v.strip())
        print(f"  [build_id_to_token] after known-text injection: {non_empty_f2} non-empty")

    return id_to_token


def train(
    model:     SpikingCausalLM,
    tokenizer: SpikeTokenizer,
    corpus:    List[str],
    qa_pairs:  List[Tuple[str, str]],
    epochs:    int,
) -> None:
    print("--- Training SpikingCausalLM ---")
    start = time.time()

    # Q/A トークン列を事前計算 + デバッグ表示
    qa_token_pairs: List[Tuple[List[int], List[int]]] = []
    id_to_token_debug = build_id_to_token(tokenizer)
    print("\n[DEBUG] Q/A token encoding:")
    for q_text, a_text in qa_pairs[:3]:  # 最初の3件だけ表示
        q_prompt = f"User: {q_text}{SEP}"
        q_ids    = tokenizer.encode(q_prompt)
        a_ids    = tokenizer.encode(a_text)
        if q_ids and a_ids:
            qa_token_pairs.append((q_ids, a_ids))
            a_decoded = [f"id={t}:'{id_to_token_debug.get(t,'?')}'" for t in a_ids[:5]]
            print(f"  Q='{q_text}' → A tokens(first5): {a_decoded}")
    # 残りのペアも追加（デバッグ表示なし）
    for q_text, a_text in qa_pairs[3:]:
        q_prompt = f"User: {q_text}{SEP}"
        q_ids    = tokenizer.encode(q_prompt)
        a_ids    = tokenizer.encode(a_text)
        if q_ids and a_ids:
            qa_token_pairs.append((q_ids, a_ids))
    print()

    for epoch in range(epochs):
        for text in corpus:
            token_ids = tokenizer.encode(text)
            if len(token_ids) > 1:
                model.train_step(token_ids, learning_rate=LR_NORMAL)

        for q_ids, a_ids in qa_token_pairs:
            model.supervised_qa_train(q_ids, a_ids, learning_rate=LR_QA, max_qa_tokens=5)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  ({time.time()-start:.1f}s)")

    print(f"Training completed in {time.time()-start:.2f}s")

    total_synapses = sum(
        sum(len(v) for v in d.values()) for d in model.weights
    )
    qa_entries = sum(len(v) for v in model.qa_weights.values())
    print(f"Total generic synapses : {total_synapses}")
    print(f"Total QA entries       : {qa_entries}")

    # QAボーナス内容を表示
    print("\n[DEBUG] QA bonus table:")
    id_to_token = build_id_to_token(tokenizer)
    for ctx_k, tok_dict in model.qa_weights.items():
        print(f"  context_key={ctx_k}:")
        for tok_id, w in sorted(tok_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"    id={tok_id:4d}  w={w:.3f}  '{id_to_token.get(tok_id, '?')}'")
    print()


def debug_top_candidates(
    model:       SpikingCausalLM,
    tokenizer:   SpikeTokenizer,
    prompt_text: str,
    id_to_token: Dict[int, str],
    top_n:       int = 10,
) -> None:
    full_prompt = f"User: {prompt_text}{SEP}"
    prompt_ids  = tokenizer.encode(full_prompt)
    qa_bonus    = model._get_qa_bonus(prompt_ids)

    model.reset_context()
    spike_history: List[List[int]] = []

    token_fan_in: Dict[int, float] = {}
    for delay_dict in model.weights:
        for s_dict in delay_dict.values():
            for t, w in s_dict.items():
                token_fan_in[t] = token_fan_in.get(t, 0.0) + w

    for tok in prompt_ids[:-1]:
        spikes   = model._get_sdr_for_token(tok)
        out      = model.transformer.forward(spikes, learning=False)
        combined = spikes + [s + model.embed_dim for s in out]
        spike_history.insert(0, combined)
        if len(spike_history) > model.max_delay + 1:
            spike_history.pop()

    last_spikes = model._get_sdr_for_token(prompt_ids[-1])
    out         = model.transformer.forward(last_spikes, learning=False)
    combined    = last_spikes + [s + model.embed_dim for s in out]
    spike_history.insert(0, combined)

    token_potentials:    Dict[int, float] = {}
    token_support_count: Dict[int, int]   = {}

    for delay, active_spikes in enumerate(spike_history):
        time_decay = max(0.1, 1.0 - delay * 0.08)
        supported: set = set()
        for s in active_spikes:
            if s in model.weights[delay]:
                for t_id, weight in model.weights[delay][s].items():
                    token_potentials[t_id] = token_potentials.get(t_id, 0.0) + weight * time_decay
                    supported.add(t_id)
        for t_id in supported:
            token_support_count[t_id] = token_support_count.get(t_id, 0) + 1

    for t_id in token_potentials:
        count = token_support_count.get(t_id, 1)
        token_potentials[t_id] *= (count ** 1.2)

    QA_SCALE = 1000.0
    if qa_bonus:
        for t_id, bonus_w in qa_bonus.items():
            extra = bonus_w * QA_SCALE
            token_potentials[t_id] = token_potentials.get(t_id, 0.0) + extra

    for t_id in list(token_potentials.keys()):
        hub = token_fan_in.get(t_id, 1.0)
        if hub > 1.0:
            token_potentials[t_id] /= math.pow(hub, 0.2)

    sorted_cands = sorted(token_potentials.items(), key=lambda x: x[1], reverse=True)
    print(f"  [DEBUG] prompt_ids={len(prompt_ids)}, qa_bonus_entries={len(qa_bonus)}, active={len(token_potentials)}")
    print(f"  [DEBUG] Top {top_n}:")
    for t_id, pot in sorted_cands[:top_n]:
        decoded = id_to_token.get(t_id, tokenizer.decode([t_id]))
        marker  = " ★QA" if t_id in qa_bonus else ""
        print(f"    id={t_id:4d}  pot={pot:9.2f}  '{decoded}'{marker}")


def generate_reply(
    model:       SpikingCausalLM,
    tokenizer:   SpikeTokenizer,
    prompt_text: str,
    id_to_token: Dict[int, str],
) -> None:
    print(f"User: '{prompt_text}'")

    full_prompt = f"User: {prompt_text}{SEP}"
    prompt_ids  = tokenizer.encode(full_prompt)

    if DEBUG:
        debug_top_candidates(model, tokenizer, prompt_text, id_to_token)

    if not prompt_ids:
        print("  [WARN] Tokenization empty.\n")
        return

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=MAX_GEN_TOKENS,
        temperature=TEMPERATURE,
        question_ids=prompt_ids,
    )

    if not generated_ids:
        print("  [WARN] No tokens generated.\n")
        return

    result_text = tokenizer.decode(generated_ids)

    if "User:" in result_text:
        result_text = result_text.split("User:")[0].strip()

    for p in ["。", "！", "？", ".", "!", "?"]:
        if p in result_text:
            result_text = result_text.split(p)[0] + p
            break

    print(f"Assistant: {result_text.strip()}\n")


def main() -> None:
    print("=== Spiking Causal LM Chat Demo v4 ===\n")

    corpus, qa_pairs = load_data(DATA_PATH, MAX_SAMPLES)
    print(f"Loaded {len(qa_pairs)} Q/A pairs.\n")

    tokenizer = SpikeTokenizer(vocab_size=VOCAB_SIZE)
    print("Training BPE Tokenizer...")
    tokenizer.train(corpus)
    actual_vocab = tokenizer.vocab_size
    print(f"Actual Vocabulary Size: {actual_vocab}\n")

    model = SpikingCausalLM(
        vocab_size=actual_vocab,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        use_lif=True,
    )
    patch_sparsity(model, SPARSITY)

    # id→token テーブルを登録（supervised_qa_train での除外判定に使用）
    id_to_token = build_id_to_token(tokenizer)
    model.register_vocab(id_to_token)

    train(model, tokenizer, corpus, qa_pairs, epochs=EPOCHS)

    # SDR衝突率確認
    all_sdrs = list(model.token_to_sdr.values())
    if len(all_sdrs) >= 2:
        sample   = all_sdrs[:min(len(all_sdrs), 20)]
        overlaps = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                s1 = set(sample[i])
                s2 = set(sample[j])
                u  = len(s1 | s2)
                if u > 0:
                    overlaps.append(len(s1 & s2) / u)
        avg_ov = sum(overlaps) / len(overlaps) if overlaps else 0.0
        print(f"[DEBUG] Avg SDR Jaccard overlap: {avg_ov:.4f}\n")

    print("--- Chat Inference ---")
    for q, _ in qa_pairs[:5]:
        generate_reply(model, tokenizer, q, id_to_token)


if __name__ == "__main__":
    main()