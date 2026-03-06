# {
#     "//": "ディレクトリパス: tests/test_snn_transformer.py",
#     "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマー・テスト",
#     "//": "ファイルの目的や内容: v2.4.0相当の SpikingTransformerModel が、行列演算なしで時系列のトークン列を学習し、予測(Generate)できるかを確認する。"
# }

from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def test_transformer_v2():
    print("--- 📚 Spiking Transformer のテストを開始 ---")
    
    # テスト用の極小設定（語彙数50）
    config = SNNTransformerConfig(vocab_size=50, embed_dim=16, num_layers=1)
    model = SpikingTransformerModel(config)
    
    # "The quick brown fox jumps over the lazy dog" 的な仮のトークン列
    input_sequence = [10, 25, 42, 18, 5, 25]
    
    print("1. 学習フェーズ (Learn Sequence) - 内部でSTP/STDP/予測符号化が走ります...")
    model.learn_sequence(input_sequence)
    print("   -> 学習完了。")
    
    print("\n2. 生成フェーズ (Generate)")
    # プロンプトとして最初の2トークンを与え、続きを予測させる
    prompt = [10, 25]
    print(f"   プロンプト: {prompt}")
    generated_ids, debug_logs = model.generate(prompt, max_length=5, debug=False)
    
    print(f"   生成されたトークン列: {generated_ids}")
    print("\n--- ✅ テスト完了：推論と学習ループがエラーなく実行されました ---")

if __name__ == "__main__":
    test_transformer_v2()