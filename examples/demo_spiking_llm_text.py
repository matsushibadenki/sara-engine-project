_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_llm_text.py",
    "//": "タイトル: スパイキングLLMの自然言語パイプラインデモ",
    "//": "目的: SpikeTokenizerを用いて自然言語(日本語/英語)を動的に学習し、SNNモデルで自己回帰テキスト生成を行うエンドツーエンドのテスト。"
}

from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

def main():
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

    # 3. Spiking LLM の初期化 (語彙サイズをトークナイザーに合わせる)
    model = SpikingLLM(
        vocab_size=tokenizer.vocab_size, 
        d_model=1024,  # <-- 128から1024へ変更！
        num_layers=2, 
        num_heads=4
    )

    # 4. モデルの学習 (テキストをエンコードして入力)
    print("\n[Training Phase]")
    epochs = 6
    for epoch in range(epochs):
        for text in training_texts:
            token_ids = tokenizer.encode(text)
            model.learn_sequence(token_ids)
        print(f" - Epoch {epoch + 1}/{epochs} completed.")

    # 5. 推論と生成 (プロンプトを与えて続きを予測させる)
    print("\n[Generation Phase]")
    
    # テスト用プロンプト (既知の文の出だし)
    prompts = [
        "Sara Engine",
        "人工知能 の",
        "SNN"
    ]

    for prompt_text in prompts:
        print(f"\nPrompt: '{prompt_text}'")
        
        # エンコード
        prompt_ids = tokenizer.encode(prompt_text)
        
        # SNNで生成 (最大5トークン)
        generated_ids = model.generate(prompt_tokens=prompt_ids, max_new_tokens=5)
        
        # デコードして結果を表示
        result_text = tokenizer.decode(generated_ids)
        print(f"Generated: '{result_text}'")

    print("\nDemo completed successfully.")

if __name__ == "__main__":
    main()