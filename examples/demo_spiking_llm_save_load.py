from sara_engine.models.spiking_llm import SpikingLLM
import json
import os
_FILE_INFO = {
    "//1": "ディレクトリパス: examples/demo_spiking_llm_save_load.py",
    "//2": "タイトル: スパイキングLLMのモデル保存・読み込みデモ",
    "//3": "目的: SpikingLLMのシナプス重みと direct_map（コンテキストマッピング）をJSONとして外部ファイルに保存し、"
           "新規インスタンスへ完全に復元するロジックを実装する。",
}


def save_spiking_llm(model: SpikingLLM, filepath: str) -> None:
    """
    モデル内の重みと direct_map をJSONシリアライズして保存。
    _direct_map: Tuple キーを文字列リストに変換して JSON 互換にする。
    """
    # _direct_map のキー (tuple) は JSON で直接使えないため、文字列リストに変換
    serializable_direct_map = {
        str(list(k)): {str(tok_id): count for tok_id, count in v.items()}
        for k, v in model._direct_map.items()
    }
    state = {
        "lm_head_w": [
            {str(k): v for k, v in row.items()} for row in model.lm_head_w
        ],
        "transformer": [
            {"ffn_w": [{str(k): v for k, v in fw.items()}
                       for fw in layer.ffn_w]}
            for layer in model.transformer.layers
        ],
        "direct_map": serializable_direct_map,
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f)


def load_spiking_llm(model: SpikingLLM, filepath: str) -> None:
    """JSONから重みと direct_map を読み込み、型を復元する。"""
    import ast

    with open(filepath, "r", encoding="utf-8") as f:
        state = json.load(f)

    # lm_head_w の復元
    model.lm_head_w = [
        {int(k): float(v) for k, v in layer.items()} for layer in state["lm_head_w"]
    ]

    # transformer.ffn_w の復元
    for i, layer_state in enumerate(state["transformer"]):
        model.transformer.layers[i].ffn_w = [
            {int(k): float(v) for k, v in fw.items()} for fw in layer_state["ffn_w"]
        ]

    # _direct_map の復元: 文字列リストキー → tuple キーに逆変換
    model._direct_map = {}
    for str_key, tok_counts in state.get("direct_map", {}).items():
        tuple_key = tuple(ast.literal_eval(str_key))
        model._direct_map[tuple_key] = {int(tok_id): float(
            count) for tok_id, count in tok_counts.items()}

    # _sdr_cache はエンコード時に再構築されるためクリアしておく
    model._sdr_cache = {}


def main() -> None:
    print("Starting Model Save/Load Demo...\n")

    workspace_dir = os.path.join(os.getcwd(), "workspace", "models")
    model_path = os.path.join(workspace_dir, "spiking_llm_weights.json")

    vocab_size = 10000
    vocab_map = {10: "I", 11: "am", 12: "learning",
                 13: "SNN", 14: "Transformer", 15: ".", 99: "[UNK]"}

    def decode(tokens: list) -> str:
        return " ".join([vocab_map.get(t, str(t)) for t in tokens])

    training_sequence = [10, 11, 12, 13, 14, 15]
    prompt = [10]

    # ==========================================
    # Phase 1: Train and Save
    # ==========================================
    print("[Phase 1: Training original model]")
    model_A = SpikingLLM(vocab_size=vocab_size, d_model=128, num_layers=2)

    for _ in range(5):
        model_A.learn_sequence(training_sequence)

    generated_A = model_A.generate(prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model A (Trained) Generation: {decode(generated_A)}")

    print(f"Saving model A state to: {model_path}")
    save_spiking_llm(model_A, model_path)
    print("Save complete.\n")

    # ==========================================
    # Phase 2: Load into a brand new model
    # ==========================================
    print("[Phase 2: Loading into a brand new model]")
    model_B = SpikingLLM(vocab_size=vocab_size, d_model=128, num_layers=2)

    generated_B_before = model_B.generate(
        prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model B (Before Load) Generation: {decode(generated_B_before)}")

    print("Loading weights...")
    load_spiking_llm(model_B, model_path)

    generated_B_after = model_B.generate(
        prompt_tokens=prompt, max_new_tokens=5)
    print(f"Model B (After Load)  Generation: {decode(generated_B_after)}")

    assert generated_A == generated_B_after, (
        f"Error: Loaded model did not reproduce the same output!\n"
        f"  Model A: {decode(generated_A)}\n"
        f"  Model B: {decode(generated_B_after)}"
    )
    print("\nSuccess! The learned synaptic weights and context mappings were perfectly restored.")


if __name__ == "__main__":
    main()
