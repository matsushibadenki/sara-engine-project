from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sara_engine.tokenization.exact_acceleration import (
    BoundedExactTokenizerAdapter,
    tokenizer_fingerprint,
)
from sara_engine.utils.tokenizer import SaraTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _trained_tokenizer(tmp_path: Path) -> SaraTokenizer:
    tokenizer = SaraTokenizer(
        vocab_size=256,
        model_path=str(tmp_path / "tokenizer.json"),
    )
    tokenizer.train(
        [" alpha beta alpha", "猫 は走る。", "神经网络处理事件。"],
        save=False,
    )
    return tokenizer


def test_bounded_exact_adapter_matches_frozen_tokenizer(tmp_path):
    tokenizer = _trained_tokenizer(tmp_path)
    adapter = BoundedExactTokenizerAdapter(
        tokenizer,
        max_entries=8,
        max_state_bytes=4096,
        max_tokens_per_entry=32,
    )

    for text in (" alpha beta alpha", "猫 は走る。", "神经网络处理事件。"):
        assert adapter.encode(text) == tokenizer.encode(text)
        assert adapter.decode(adapter.encode(text)) == tokenizer.decode(
            tokenizer.encode(text)
        )
    assert adapter.stats()["hits"] > 0
    assert adapter.stats()["state_bytes"] <= 4096


def test_bounded_exact_adapter_evicts_and_bypasses_large_entries(tmp_path):
    tokenizer = _trained_tokenizer(tmp_path)
    adapter = BoundedExactTokenizerAdapter(
        tokenizer,
        max_entries=2,
        max_state_bytes=1024,
        max_tokens_per_entry=2,
    )

    for text in ("alpha", "beta", "猫", "神经"):
        adapter.encode(text)
    adapter.encode("unknown-long-token")

    stats = adapter.stats()
    assert stats["entries"] <= 2
    assert stats["state_bytes"] <= 1024
    assert stats["evictions"] > 0
    assert stats["bypassed"] > 0


def test_bounded_exact_adapter_rejects_malformed_utf8(tmp_path):
    adapter = BoundedExactTokenizerAdapter(_trained_tokenizer(tmp_path))

    with pytest.raises(UnicodeDecodeError):
        adapter.encode_utf8(b"\xff")


def test_bounded_exact_adapter_keeps_snapshot_after_source_changes(tmp_path):
    tokenizer = _trained_tokenizer(tmp_path)
    adapter = BoundedExactTokenizerAdapter(tokenizer)
    before = adapter.encode("new")
    fingerprint = adapter.fingerprint

    tokenizer._add_token("new")

    assert adapter.encode("new") == before
    assert adapter.fingerprint == fingerprint
    assert tokenizer_fingerprint(tokenizer) != fingerprint


def test_tokenizer_training_can_skip_model_write(tmp_path):
    model_path = tmp_path / "not-written.json"
    tokenizer = SaraTokenizer(vocab_size=64, model_path=str(model_path))

    tokenizer.train(["alpha beta"], save=False)

    assert model_path.exists() is False


def test_phase27_tokenizer_benchmark_passes():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase27_tokenizer_acceleration_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase27_tokenizer_acceleration_benchmark", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.build_report(module.load_cases(module.DEFAULT_FIXTURE))

    assert report["passed"] is True
    assert report["production_path_changed"] is False
    if report["rust_scalar_reference_available"]:
        assert report["rust_path_observed"] is True
        assert report["rust_scalar_reference_equivalent"] is True
        assert all(
            case["rust_scalar_token_ids_equivalent"] is True
            for case in report["cases"].values()
        )
    else:
        assert report["rust_path_observed"] is False
    assert report["gigatoken_path_observed"] is False
    assert report["checks"]["decode_round_trip_preserved"] is True
    assert report["checks"]["snapshot_state_bounded"] is True
    assert report["checks"]["peak_rss_growth_bounded"] is True
    measurement = report["resource_measurement"]
    assert measurement["resource_trace_count"] > 0
    assert measurement["snapshot_state_bytes"] > 0
    if report["rust_scalar_reference_available"]:
        assert report["checks"]["equal_trace_rust_outputs_equivalent"] is True
        assert report["checks"]["rust_warm_replay_equivalent"] is True
        assert report["checks"]["downstream_replay_equivalent"] is True
        assert report["checks"]["rust_boundary_calls_accounted"] is True
        assert measurement["rust_boundary_calls"] == 2 * measurement["resource_trace_count"]
        assert report["rust_batch_reference_available"] is True
        assert report["rust_batch_reference_equivalent"] is True
        assert report["checks"]["rust_batch_warm_replay_equivalent"] is True
        assert report["checks"]["rust_batch_downstream_replay_equivalent"] is True
        assert report["checks"]["rust_batch_boundary_reduced"] is True
        assert measurement["rust_batch_boundary_calls"] == 2
        assert report["checks"]["large_trace_batch_equivalent"] is True
        assert report["checks"]["repeated_median_samples_complete"] is True
        assert measurement["median_trace_count"] == 300
        assert measurement["median_repetitions"] == 7
        assert report["rust_snapshot_reference_available"] is True
        assert report["rust_snapshot_reference_equivalent"] is True
        assert report["checks"]["large_trace_snapshot_equivalent"] is True
        assert report["checks"]["rust_snapshot_downstream_replay_equivalent"] is True
        assert measurement["rust_snapshot_boundary_calls"] == 2
