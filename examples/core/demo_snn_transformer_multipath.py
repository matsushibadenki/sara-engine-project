from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
import sys
import os
_FILE_INFO = {
    "path": "examples/demo_snn_transformer_multipath.py",
    "title": "スパイキング・トランスフォーマー Multi-Pathway デモ",
    "purpose": "構築したMulti-Pathway SNN Transformerの動作確認。多言語テキストを学習し、workspaceディレクトリにモデルとログを保存する。"
}


# Add src to Python path if running from root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))


def main():
    # Setup workspace directory
    workspace_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'workspace', 'transformer_demo'))
    os.makedirs(workspace_dir, exist_ok=True)

    print("[INFO] Initializing Spiking Transformer with Multi-Pathway Attention...")
    # Configure with 4 pathways (equivalent to 4 Attention Heads)
    config = SNNTransformerConfig(
        vocab_size=256, embed_dim=128, num_layers=2, num_pathways=4)
    model = SpikingTransformerModel(config)

    # Multilingual training data (English & Japanese)
    training_data = [
        "The quick brown fox jumps over the lazy dog.",
        "SARA Engine is the next generation neuromorphic AI.",
        "こんにちは、世界。これはスパイキングニューラルネットワークのテストです。",
        "Matrix multiplication is no longer needed.",
        "誤差逆伝播法を使わずに、STDPのみで学習を完了させます。"
    ]

    # 英語文字レベルの連想学習には10エポック以上必要
    num_epochs = 10
    print(
        f"[INFO] Starting STDP learning on {len(training_data)} sequences x {num_epochs} epochs...")
    for epoch in range(num_epochs):
        for i, seq in enumerate(training_data):
            if epoch == 0:
                print(f"  -> Learning sequence {i+1}: {seq[:30]}...")
            model.learn_sequence(seq)
        print(f"  [Epoch {epoch + 1}/{num_epochs}] completed.")

    # Test generation
    print("\n[INFO] Testing generation capabilities...")
    prompts = [
        "The quick brown",
        "SARA Engine is",
        "こんにちは、世界"
    ]

    for prompt in prompts:
        generated = model.generate(prompt, max_length=30)
        print(f"\nPrompt: '{prompt}'")
        print(f"Result: '{generated}'")

    # Save model to workspace
    save_path = os.path.join(workspace_dir, "multi_pathway_model")
    print(f"\n[INFO] Saving model to {save_path}...")
    model.save_pretrained(save_path)
    print("[INFO] Demo completed successfully.")


if __name__ == "__main__":
    main()
