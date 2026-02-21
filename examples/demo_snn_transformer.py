_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_transformer.py",
    "//": "ファイルの日本語タイトル: SNN トランスフォーマー デモ",
    "//": "ファイルの目的や内容: UTF-8対応のSNNを訓練し、多言語テキストの自己回帰生成とモデル保存をテストする。"
}

import sys
import os

# プロジェクトルートの 'src' ディレクトリを正確にパスへ追加
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def main():
    print("="*60)
    print("SNN Transformer (Biologically Inspired LLM Alternative)")
    print("="*60)

    # 1. モデルの初期化
    config = SNNTransformerConfig(vocab_size=256, embed_dim=128, num_layers=2, ffn_dim=256)
    model = SpikingTransformerModel(config)

    # 2. Hebbian STDP による多言語学習
    print("\n[Phase 1] Training SNN with Multilingual Corpus via STDP...")
    corpus = [
        "Spiking Neural Networks represent the future of AI.",
        "SNNs consume significantly less energy than traditional ANNs.",
        "スパイキングニューラルネットワークはAIの未来を象徴しています。",
        "誤差逆伝播法を使わず、局所的なSTDPのみで学習します。"
    ]
    
    epochs = 20
    for epoch in range(epochs):
        for sentence in corpus:
            model.learn_sequence(sentence)
    print("Training complete. Spatiotemporal patterns encoded in Reservoir.")

    # 3. 推論（多言語テキスト生成）
    print("\n[Phase 2] Multilingual Text Generation...")
    
    # 英語の生成テスト (十分な長さのバイト数を指定)
    output_en = model.generate("Spiking Neural ", max_length=50)
    print(f"Result (EN): {output_en}")
    
    # 日本語の生成テスト (マルチバイトに対応するため max_length を 100 に拡張)
    output_jp = model.generate("誤差逆伝播法を", max_length=100)
    print(f"Result (JP): {output_jp}")

    # 4. ワークスペースへの保存と読み込み
    print("\n[Phase 3] Saving and Loading Model...")
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'snn-transformer-v1'))
    
    model.save_pretrained(workspace_dir)
    
    loaded_model = SpikingTransformerModel.from_pretrained(workspace_dir)
    print("Model successfully loaded from workspace.")
    
    print("\nAll tasks completed successfully. SNN Architecture is fully functional!")

if __name__ == "__main__":
    main()