import pytest


def test_rust_core_exports_expected_sparse_runtime_symbols():
    sara_rust_core = pytest.importorskip(
        "sara_engine.sara_rust_core",
        reason="Rust extension is optional unless the package is built with maturin.",
    )

    expected_symbols = {
        "calculate_sdr_overlap",
        "sparse_propagate_threshold",
        "build_direct_synapses",
        "batch_tokens_to_sdr",
        "apply_homeostatic_scaling",
        "tokenize_sara_bpe_pretokens",
        "SpikeEngine",
        "SpikeWTARouter",
        "LIFNetwork",
        "CausalSynapses",
        "ScalableSDRMemory",
        "RewardModulatedSTDP",
    }

    missing = sorted(name for name in expected_symbols if not hasattr(sara_rust_core, name))
    assert missing == []


def test_rust_sparse_propagation_accepts_supported_python_weight_shapes():
    sara_rust_core = pytest.importorskip(
        "sara_engine.sara_rust_core",
        reason="Rust extension is optional unless the package is built with maturin.",
    )

    assert sara_rust_core.sparse_propagate_threshold(
        [0], [{1: 0.7, 2: 0.2}], 4, 0.5
    ) == [1]
    assert sara_rust_core.sparse_propagate_threshold(
        [0], [[(1, 0.4), (3, 0.9)]], 4, 0.5
    ) == [3]
    assert sara_rust_core.sparse_propagate_threshold(
        [0], [[0.1, 0.8, 0.2]], 3, 0.5
    ) == [1]


def test_rust_core_rejects_invalid_runtime_parameters():
    sara_rust_core = pytest.importorskip(
        "sara_engine.sara_rust_core",
        reason="Rust extension is optional unless the package is built with maturin.",
    )

    with pytest.raises(ValueError):
        sara_rust_core.SpikeEngine(decay_rate=1.5)
    with pytest.raises(ValueError):
        sara_rust_core.SpikeWTARouter(4, 2, 3)
    with pytest.raises(ValueError):
        sara_rust_core.LIFNetwork(-0.1, 1.0)
    with pytest.raises(ValueError):
        sara_rust_core.ScalableSDRMemory(1.5)
    with pytest.raises(ValueError):
        sara_rust_core.RewardModulatedSTDP(2, 1.5)
    with pytest.raises(ValueError):
        sara_rust_core.build_direct_synapses([1, 2, 3], 0)
    with pytest.raises(ValueError):
        sara_rust_core.batch_tokens_to_sdr([[1]], 0, 0.1, 1)
    with pytest.raises(ValueError):
        sara_rust_core.apply_homeostatic_scaling([{}], [float("nan")], 1.0, 0.1)
    with pytest.raises(ValueError):
        sara_rust_core.tokenize_sara_bpe_pretokens(
            ["ab"], {"<unk>": 1}, [("a", "b"), ("a", "b")], 1
        )


def test_rust_scalar_bpe_uses_python_defined_pretoken_boundaries():
    sara_rust_core = pytest.importorskip(
        "sara_engine.sara_rust_core",
        reason="Rust extension is optional unless the package is built with maturin.",
    )

    assert sara_rust_core.tokenize_sara_bpe_pretokens(
        ["abab", " 日本", "x"],
        {
            "<unk>": 1,
            "a": 7,
            "b": 8,
            "ab": 9,
            " ": 10,
            "日": 11,
            "本": 12,
            " 日本": 13,
        },
        [("a", "b"), (" ", "日"), (" 日", "本")],
        1,
    ) == [9, 9, 13, 1]


def test_spike_attention_uses_explicit_python_fallback_for_missing_rust_attention():
    from sara_engine.core.attention import SpikeAttention

    attention = SpikeAttention(input_size=8, hidden_size=16, use_rust=True)

    assert attention.use_rust is False
    assert "Python sparse fallback" in attention.fallback_reason
