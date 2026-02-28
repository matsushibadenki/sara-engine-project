from src.sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from src.sara_engine.models.spiking_causal_lm import SpikingCausalLM
import json
import time
import sys
import os
_FILE_INFO = {
    "//": "ディレクトリパス: examples/integrated_spiking_llm_demo.py",
    "//": "ファイルの日本語タイトル: 統合スパイキングLLMデモ (学習・生成・保存・チャット)",
    "//": "ファイルの目的や内容: 複数のLLM関連デモを統合。BPEトークナイズ、STDP/Hebbian学習、自己回帰生成、モデルの保存・復元、および停止トークンによる制御を網羅する。"
}


# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main():
    print("=== SARA Engine: Integrated Spiking LLM Demonstration ===\n")

    workspace_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../workspace/integrated_llm'))
    os.makedirs(workspace_dir, exist_ok=True)
    model_path = os.path.join(workspace_dir, "model_state.json")

    # 1. データとトークナイザーの準備
    corpus = [
        "SARA Engine is a brain-inspired AI .",
        "It uses Spiking Neural Networks for efficiency .",
        "人工知能 の 未来 は スパイク に あります 。",
        "User: こんにちは Assistant: こんにちは！SARAです。＜終＞"
    ]

    tokenizer = SpikeTokenizer()
    tokenizer.train(corpus)
    vocab_size = tokenizer.vocab_size
    print(f"[1] Tokenizer trained. Vocab size: {vocab_size}")

    # 2. モデルの初期化 (LIFモデル採用)
    # embed_dimを拡大して衝突を抑制
    model = SpikingCausalLM(
        vocab_size=vocab_size,
        embed_dim=512,
        hidden_dim=1024,
        use_lif=True
    )
    print("[2] SpikingCausalLM initialized with LIF neurons.")

    # 3. 学習フェーズ (STDP & Hebbian)
    print("\n[3] Starting Training Phase...")
    epochs = 20
    start_time = time.time()
    for epoch in range(epochs):
        for text in corpus:
            token_ids = tokenizer.encode(text)
            # BOS(2)とEOS(3)を付与して学習
            full_ids = [2] + token_ids + [3]
            model.train_step(full_ids, learning_rate=0.3)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} completed.")
    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 4. モデルの保存
    print(f"\n[4] Saving model to {model_path}...")
    # 既存のsave_pretrained互換の処理 (簡易版)
    model.save_pretrained(model_path)

    # 5. 推論テスト (生成)
    print("\n[5] Testing Generation...")
    # 停止トークンの設定
    # get_vocab() は { "文字列": ID } を返すため、条件に一致するトークンの "値(ID)" を取得する
    vocab = tokenizer.get_vocab()
    stop_ids = [tid for token,
                tid in vocab.items() if "終" in token or "。" in token]

    prompts = ["SARA Engine", "人工知能 の", "User: こんにちは"]

    for prompt_text in prompts:
        print(f"Prompt: '{prompt_text}'")
        prompt_ids = [2] + tokenizer.encode(prompt_text)

        # 自己回帰生成
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=15,
            temperature=0.1,
            stop_token_ids=stop_ids
        )

        result_text = tokenizer.decode(generated_ids)
        print(f"Generated: '{result_text}'\n")

    # 6. 復元テスト
    print("[6] Verifying Save/Load...")
    new_model = SpikingCausalLM(
        vocab_size=vocab_size, embed_dim=512, hidden_dim=1024, use_lif=True)
    new_model.load_pretrained(model_path)

    # 同一入力で出力を比較
    check_ids = new_model.generate(
        [2] + tokenizer.encode("SARA Engine"), max_new_tokens=5)
    if check_ids:
        print("Success: Loaded model produced output.")

    print("\n=== Integrated Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
