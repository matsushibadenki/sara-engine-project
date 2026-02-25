from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
_FILE_INFO = {
    "//1": "ディレクトリパス: examples/demo_spiking_llm_text.py",
    "//2": "タイトル: スパイキングLLMの自然言語パイプラインデモ",
    "//3": "目的: SpikeTokenizerを用いて自然言語(日本語/英語)を動的に学習し、SNNモデルで自己回帰テキスト生成を行うエンドツーエンドのテスト。"
}


def main() -> None:
    print("Starting End-to-End Spiking LLM with Text Tokenizer...\n")

    # 1. 訓練データの準備 (多言語対応のテスト)
    training_texts = [
        "SNN is highly energy efficient .",
        "Sara Engine uses local Hebbian learning .",
        "人工知能 の 未来 は スパイク に あります 。"
    ]

    # 2. トークナイザーの初期化と学習
    tokenizer = SpikeTokenizer()
    tokenizer.train(training_texts)
    print(f"Vocabulary built. Total vocab size: {tokenizer.vocab_size}")

    # 3. Spiking LLM の初期化
    # 改善: d_model=128（語彙サイズ24に適したSDR次元数）
    model = SpikingLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4
    )

    # 4. モデルの学習
    # 改善: エポック数を 40 → 100 に増加して十分に収束させる
    print("\n[Training Phase]")
    epochs = 100
    for epoch in range(epochs):
        for text in training_texts:
            token_ids = tokenizer.encode(text)
            model.learn_sequence(token_ids)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f" - Epoch {epoch + 1}/{epochs} completed.")

    # 5. 推論と生成 (プロンプトを与えて続きを予測させる)
    print("\n[Generation Phase]")

    # テスト用プロンプト
    # 注: スペース区切りの文として入力すると学習語彙と一致したトークン化が行われる
    prompts = [
        "Sara Engine",       # 学習文: "Sara Engine uses local Hebbian learning ."
        "人工知能 の",         # 学習文: "人工知能 の 未来 は スパイク に あります 。"
        "SNN is"             # 学習文: "SNN is highly energy efficient ."
    ]

    for prompt_text in prompts:
        print(f"\nPrompt: '{prompt_text}'")

        # エンコード
        prompt_ids = tokenizer.encode(prompt_text)

        # SNNで生成 (最大5トークン, top_k=1 で確定的生成)
        generated_ids = model.generate(
            prompt_tokens=prompt_ids,
            max_new_tokens=5,
            top_k=1,
            temperature=0.1
        )

        # デコードして結果を表示
        result_text = tokenizer.decode(generated_ids)
        print(f"Generated: '{result_text}'")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
