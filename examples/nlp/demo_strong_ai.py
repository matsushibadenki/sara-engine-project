from sara_engine.pipelines.text_generation import pipeline as text_pipeline
from sara_engine.auto import AutoTokenizer, AutoStrongSpikingLM
import os
{
    "//": "ディレクトリパス: examples/nlp/demo_strong_ai.py",
    "//": "ファイルの日本語タイトル: 強いAI (階層的予測符号化) デモ",
    "//": "ファイルの目的や内容: 予測符号化を用いたStrongSpikingLMの学習と、深い文脈からの推論テスト。"
}


def main():
    workspace_dir = os.path.join(os.path.dirname(
        __file__), "..", "..", "workspace", "strong_ai_demo")
    os.makedirs(workspace_dir, exist_ok=True)

    print("=== SARA Engine: Strong AI (Hierarchical Predictive Coding) Demo ===")

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(workspace_dir)

    corpus = [
        "SARA Engine is an artificial intelligence based on Spiking Neural Networks.",
        "It does not use backpropagation or matrix multiplication.",
        "This makes SARA highly energy efficient and biologically plausible."
    ]

    print("1. Training BPE Tokenizer...")
    tokenizer.train(corpus)

    print("\n2. Initializing StrongSpikingLM (Brain-inspired Architecture)...")
    # 階層的予測符号化モデルの読み込み
    model = AutoStrongSpikingLM.from_pretrained(workspace_dir)
    strong_pipeline = text_pipeline(
        "text-generation", model=model, tokenizer=tokenizer)

    print("3. Autonomous Learning via Predictive Error (Surprisal)...")
    # 教師なしで自律的に文脈と表現を学習
    for _ in range(10):  # 予測誤差が収束するまで反復
        for text in corpus:
            strong_pipeline.learn(text)

    print("\n4. Testing Generative Inference...")
    prompts = [
        "SARA Engine is an",
        "It does not use",
        "This makes SARA highly"
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = strong_pipeline(prompt, max_new_tokens=10)
        print(f"Generated: '{generated}'")

    strong_pipeline.save_pretrained(workspace_dir)
    print(f"\nModel autonomously evolved and saved to {workspace_dir}")


if __name__ == "__main__":
    main()
