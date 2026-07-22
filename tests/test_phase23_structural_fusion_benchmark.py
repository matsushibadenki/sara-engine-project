from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase23_structural_fusion_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase23_structural_fusion_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase23_structural_fusion_benchmark_passes():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "phase23_structural_fusion_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))

    assert report["passed"] is True
    assert report["metrics"]["decision_accuracy"] == 1.0
    assert report["checks"]["durable_mutation_blocked"] is True
    assert report["checks"]["cross_modal_hypothesis_boundary"] is True
    assert report["checks"]["cross_modal_hypothesis_contradiction_freeze"] is True
    assert report["checks"]["risa_hypothesis_remains_unverified"] is True
    assert report["checks"]["modality_dropout_is_symmetric"] is True
    assert report["checks"]["asynchronous_window_boundary"] is True
    assert report["checks"]["verified_bundle_episode_and_subgraph_bridge"] is True
    assert report["checks"]["rejected_bundle_bridge_isolation"] is True


def test_phase23_structural_fusion_rejects_contradiction():
    module = _load_module()
    result = module.MultimodalStructuralVerifier().verify(
        (
            module.ModalityEvidence("vision", "open", 1, "vision-ref"),
            module.ModalityEvidence("audio", "closed", 2, "audio-ref"),
        ),
        expected_modalities=("vision", "audio"),
    )

    assert result.decision == "abstain_cross_modal_contradiction"
    assert result.durable_mutation_allowed is False
