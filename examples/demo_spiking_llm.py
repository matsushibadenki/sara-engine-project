from sara_engine.models.spiking_llm import SpikingLLM
import json
import os
_FILE_INFO = {
    "//1": "ディレクトリパス: examples/demo_spiking_llm.py",
    "//2": "タイトル: スパイキングLLMの学習と生成デモ",
    "//3": "目的: SpikingLLMを用いて、短いトークン系列の学習(Hebbian)と、その直後の自己回帰的なテキスト生成推論の動作をテストする。"
}


def main():
    print("Starting Spiking LLM Demo (Training & Generation)...")

    workspace_dir = os.path.join(os.getcwd(), "workspace", "logs")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "spiking_llm_log.json")

    # Initialize Model
    vocab_size = 10000
    model = SpikingLLM(vocab_size=vocab_size, d_model=128,
                       num_layers=2, num_heads=4)

    # Vocabulary mapping for demonstration purposes
    vocab_map = {
        10: "I",
        11: "am",
        12: "learning",
        13: "SNN",
        14: "Transformer",
        15: ".",
        99: "[UNK]"
    }

    def decode(tokens):
        return " ".join([vocab_map.get(t, str(t)) for t in tokens])

    # Training Sequence: "I am learning SNN Transformer ."
    training_sequence = [10, 11, 12, 13, 14, 15]
    print("\n[Training Phase]")
    print(f"Target Sentence: {decode(training_sequence)}")

    # Multiple epochs to strengthen Hebbian connections
    epochs = 5
    for epoch in range(epochs):
        model.learn_sequence(training_sequence)
        print(f" - Epoch {epoch + 1}/{epochs} completed.")

    # Generation Phase
    print("\n[Generation Phase]")
    # Give the model the first word "I" and ask it to generate the rest
    prompt = [10]
    print(f"Prompt: {decode(prompt)}")

    generated_tokens = model.generate(prompt_tokens=prompt, max_new_tokens=5)

    print(f"Generated Result: {decode(generated_tokens)}")

    # Save logs
    logs = {
        "event": "Spiking LLM Training and Generation",
        "training_sequence": training_sequence,
        "epochs": epochs,
        "prompt": prompt,
        "generated_sequence": generated_tokens
    }
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    print(f"\nDemo completed. Logs saved to: {log_file_path}")


if __name__ == "__main__":
    main()
