# examples/demo_spiking_function_calling.py
# スパイキング因果LLM 関数呼び出し（Function Calling）デモ
# 目的: BPEのスペース挿入をバイパスし、確実にPythonブリッジを検知して結果を差し戻す。

from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.models.spiking_causal_lm import SpikingCausalLM
import re
import random
from typing import List, Dict, Tuple

_FILE_INFO = {
    "path":  "examples/demo_spiking_function_calling.py",
    "title": "スパイキング因果LLM Function Calling デモ",
    "description": "BPEのスペース挿入をバイパスし、確実にPythonブリッジを検知して結果を差し戻す。"
}

VOCAB_SIZE     = 500
EMBED_DIM      = 256
HIDDEN_DIM     = 512
SPARSITY       = 0.10
EPOCHS         = 30
LR_NORMAL      = 0.3
LR_QA          = 2.0
SEP            = " Assistant: "
EOS_WORD       = "＜終＞"

def build_id_to_token(tokenizer: SpikeTokenizer) -> Dict[int, str]:
    id_to_token: Dict[int, str] = {}
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        candidate: Dict[int, str] = {v: k for k, v in vocab.items()}
        non_empty_c = sum(1 for v in candidate.values() if v.strip())
        if non_empty_c > len(candidate) * 0.3:
            return candidate

    for i in range(tokenizer.vocab_size):
        try:
            text = tokenizer.decode([i])
            id_to_token[i] = text if text is not None else ""
        except Exception:
            id_to_token[i] = ""
    return id_to_token

def train_agent(
    model:     SpikingCausalLM,
    tokenizer: SpikeTokenizer,
    corpus:    List[str],
    qa_pairs:  List[Tuple[str, str]],
    epochs:    int,
) -> None:
    print("--- Training Agentic SNN ---")
    qa_token_pairs: List[Tuple[List[int], List[int]]] = []
    
    for q_text, a_text in qa_pairs:
        q_prompt = f"User: {q_text}{SEP}"
        q_ids    = tokenizer.encode(q_prompt)
        a_ids    = tokenizer.encode(a_text + EOS_WORD)
        if q_ids and a_ids:
            qa_token_pairs.append((q_ids, a_ids))

    for epoch in range(epochs):
        for text in corpus:
            token_ids = tokenizer.encode(text)
            if len(token_ids) > 1:
                model.train_step(token_ids, learning_rate=LR_NORMAL)

        for q_ids, a_ids in qa_token_pairs:
            model.supervised_qa_train(q_ids, a_ids, learning_rate=LR_QA, max_qa_tokens=20)
            
    print("Training completed.\n")

def execute_function(func_str: str) -> str:
    """SNNが要求した関数（計算）をPythonでフックして実行する"""
    print(f"  [Python Bridge] 計算要求を検知: {func_str}")
    try:
        if func_str:
            result = eval(func_str)
            print(f"  [Python Bridge] 計算結果: {result}")
            return str(result)
        return "Error"
    except Exception:
        return "Error"

def generate_agentic_reply(
    model:       SpikingCausalLM,
    tokenizer:   SpikeTokenizer,
    user_text:   str,
    id_to_token: Dict[int, str],
) -> None:
    print(f"User: '{user_text}'")
    
    context_text = f"User: {user_text}{SEP}"
    prompt_ids = tokenizer.encode(context_text)
    
    stop_ids = [k for k, v in id_to_token.items() if "終" in v]
    
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=25,
        temperature=0.01,
        question_ids=prompt_ids,
        stop_token_ids=stop_ids,
        repetition_penalty=0.01,
        repetition_window=3
    )
    
    raw_result_text = tokenizer.decode(generated_ids)
    normalized_text = raw_result_text.replace(" ", "")
    
    if "CALC" in normalized_text:
        # BPEの揺れに負けないよう、CALCの直後にある数字と演算子だけを確実に抽出する
        match = re.search(r'CALC[＞>]*([0-9\+\-\*\/\.\(\)]+)', normalized_text)
        if match:
            func_part = match.group(1).strip()
            
            # CALCの前の文章（「計算します。」など）を抽出
            prefix_match = re.search(r'(.*?)＜?CALC', normalized_text)
            prefix = prefix_match.group(1) if prefix_match else ""
            
            calc_result = execute_function(func_part)
            
            # コンテキストに結果を差し戻して 2nd Pass（続きの推論）へ
            agent_thought = f"{prefix} ＜CALC＞ {func_part} ＜/CALC＞ {calc_result} "
            new_context = context_text + agent_thought
            new_prompt_ids = tokenizer.encode(new_context)
            
            second_gen_ids = model.generate(
                new_prompt_ids,
                max_new_tokens=15,
                temperature=0.01,
                question_ids=prompt_ids,
                stop_token_ids=stop_ids,
            )
            
            second_text = tokenizer.decode(second_gen_ids).replace(" ", "")
            final_raw_text = agent_thought + second_text
        else:
            final_raw_text = normalized_text
    else:
        final_raw_text = normalized_text

    # 出力のクリーニング（ユーザーに見やすく整形）
    clean_text = final_raw_text.replace("User:", "").replace("Assistant:", "")
    clean_text = clean_text.replace("＜終＞", "").replace("終", "")
    
    # 計算過程の視覚的フォーマットと不要なタグの除去
    clean_text = clean_text.replace("＜CALC＞", " [計算: ").replace("＜/CALC＞", "] ->")
    clean_text = clean_text.replace("＜", "").replace("＞", "")
    
    # 句読点の後に少しだけスペースを入れて綺麗に
    clean_text = clean_text.replace("。", "。 ").replace("！", "！ ")
    
    # 連続するスペースを圧縮
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    print(f"Assistant: {clean_text}\n")


def main() -> None:
    print("=== Spiking Causal LM Agentic / Function Calling Demo ===\n")

    qa_pairs = [
        ("こんにちは", "こんにちは！SARAです。何かお手伝いしましょうか？"),
        ("10足す20は？", "計算します。＜CALC＞10+20＜/CALC＞"),
        ("10足す20は？ Assistant: 計算します。 ＜CALC＞ 10+20 ＜/CALC＞ 30", "です。計算完了しました。"),
        ("3掛ける4は？", "計算します。＜CALC＞3*4＜/CALC＞"),
        ("3掛ける4は？ Assistant: 計算します。 ＜CALC＞ 3*4 ＜/CALC＞ 12", "になります。"),
        ("あなたは誰？", "私はSARAです。外部ツールと連携できるSNNのAIです。"),
    ]
    
    corpus = [f"User: {q}{SEP}{a}{EOS_WORD}" for q, a in qa_pairs]

    tokenizer = SpikeTokenizer(vocab_size=VOCAB_SIZE)
    print("Training BPE Tokenizer...")
    tokenizer.train(corpus)

    model = SpikingCausalLM(
        vocab_size=tokenizer.vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, use_lif=True
    )
    
    def patched_sdr(token_id: int, sparsity: float = SPARSITY) -> List[int]:
        if token_id not in model.token_to_sdr:
            random.seed(token_id)
            model.token_to_sdr[token_id] = random.sample(range(model.embed_dim), max(1, int(model.embed_dim * sparsity)))
            random.seed()
        return model.token_to_sdr[token_id]
    model._get_sdr_for_token = patched_sdr

    id_to_token = build_id_to_token(tokenizer)
    model.register_vocab(id_to_token)

    train_agent(model, tokenizer, corpus, qa_pairs, epochs=EPOCHS)

    print("--- Function Calling Inference ---")
    test_queries = [
        "こんにちは",
        "10足す20は？",
        "3掛ける4は？",
        "あなたは誰？"
    ]
    
    for q in test_queries:
        generate_agentic_reply(model, tokenizer, q, id_to_token)


if __name__ == "__main__":
    main()