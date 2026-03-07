# {
#     "//": "ディレクトリパス: tests/test_snn_transformer.py",
#     "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマー・テスト",
#     "//": "ファイルの目的や内容: v2.4.0相当の SpikingTransformerModel が、行列演算なしで時系列のトークン列を学習し、予測(Generate)できるかを確認する。"
# }

from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig, NGramSpikeGenerator

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


def test_inference_fallback_linear_readout_prevents_silence():
    config = SNNTransformerConfig(vocab_size=20, embed_dim=8, num_layers=1)
    model = SpikingTransformerModel(config)

    prompt = [5]
    spikes = NGramSpikeGenerator.generate_spikes(prompt, model.num_ngram_levels, model.reservoir_size)
    target_id = 8
    for s in spikes:
        model.readout_synapses[s][target_id] = (0.1, 0)

    generated, logs = model.generate(prompt, max_length=1, temperature=0.0, fire_threshold=0.4, debug=True)

    assert len(generated) == 2
    assert generated[-1] == target_id
    assert logs
    assert logs[0].get("stop_reason") in ("fallback_linear_readout", "")


def test_prompt_warmup_does_not_poison_homeostatic_thresholds():
    config = SNNTransformerConfig(vocab_size=20, embed_dim=8, num_layers=1)
    model = SpikingTransformerModel(config)

    warmup_token = 5
    current_token = 6
    target_id = 7

    warmup_spikes = NGramSpikeGenerator.generate_spikes([warmup_token], model.num_ngram_levels, model.reservoir_size)
    current_spikes = NGramSpikeGenerator.generate_spikes([current_token, warmup_token], model.num_ngram_levels, model.reservoir_size)

    for s in warmup_spikes:
        model.readout_synapses[s][target_id] = (0.35, 0)
    for s in current_spikes:
        model.readout_synapses[s][target_id] = (0.22, 0)

    generated, _logs = model.generate(
        [warmup_token, current_token],
        max_length=1,
        temperature=0.0,
        fire_threshold=0.4,
        debug=True,
    )

    assert generated[-1] == target_id


def test_generate_stops_before_appending_eos():
    config = SNNTransformerConfig(vocab_size=20, embed_dim=8, num_layers=1)
    model = SpikingTransformerModel(config)

    prompt = [5]
    spikes = NGramSpikeGenerator.generate_spikes(prompt, model.num_ngram_levels, model.reservoir_size)
    eos_id = 3
    for s in spikes:
        model.readout_synapses[s][eos_id] = (1.0, 0)

    generated, logs = model.generate(prompt, max_length=1, temperature=0.0, fire_threshold=0.3, debug=True)

    assert generated == prompt
    assert logs


def test_generate_blocks_repeated_ngram_loops():
    config = SNNTransformerConfig(vocab_size=32, embed_dim=8, num_layers=1)
    model = SpikingTransformerModel(config)

    prompt = [5, 6, 7]
    seq_a = NGramSpikeGenerator.generate_spikes([7, 6, 5], model.num_ngram_levels, model.reservoir_size)
    seq_b = NGramSpikeGenerator.generate_spikes([8, 7, 6, 5], model.num_ngram_levels, model.reservoir_size)

    for s in seq_a:
        model.readout_synapses[s][8] = (1.2, 0)
    for s in seq_b:
        model.readout_synapses[s][5] = (1.2, 0)

    generated, logs = model.generate(prompt, max_length=6, temperature=0.0, fire_threshold=0.3, debug=True)

    assert generated[:4] == [5, 6, 7, 8]
    assert len(generated) <= 5

if __name__ == "__main__":
    test_transformer_v2()
