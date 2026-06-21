import importlib.util
import os


def _load_ladder_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "real_data_external_validity_ladder.py")
    )
    spec = importlib.util.spec_from_file_location("real_data_external_validity_ladder_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_corpus(path, count: int = 32) -> str:
    lines = []
    for index in range(count):
        lines.append(
            " ".join(
                [
                    f"topic{index}",
                    f"anchor{index}",
                    f"signal{index}",
                    f"memory{index}",
                    "sparse",
                    "event",
                    "retrieval",
                    "energy",
                    "efficient",
                    f"detail{index}",
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_parse_profile_specs_accepts_custom_ladder():
    module = _load_ladder_module()

    profiles = module.parse_profile_specs(["tiny:12:4", "pilot:24:8"])

    assert profiles == [
        {"name": "tiny", "max_docs": 12, "max_cases": 4},
        {"name": "pilot", "max_docs": 24, "max_cases": 8},
    ]


def test_real_data_external_validity_ladder_passes_on_sparse_fixture(tmp_path):
    module = _load_ladder_module()
    corpus_path = _write_corpus(tmp_path / "corpus.txt", count=48)

    report = module.run_real_data_external_validity_ladder(
        corpus_path=corpus_path,
        profiles=[
            {"name": "tiny", "max_docs": 12, "max_cases": 4},
            {"name": "pilot", "max_docs": 24, "max_cases": 8},
        ],
        regression_tolerance=0.05,
        update_history=False,
    )

    assert report["suite_name"] == "RealDataExternalValidityLadder"
    assert report["passed"] is True
    assert report["checks"]["all_profiles_passed"] is True
    assert report["checks"]["scale_doc_counts_monotonic"] is True
    assert report["metrics"]["profile_count"] == 2
    assert report["metrics"]["passed_profile_count"] == 2
    assert report["metrics"]["min_real_data_qa_accuracy"] == 1.0
    assert report["metrics"]["min_ann_cost_advantage_proxy"] >= 2.0
    assert report["metrics"]["min_negative_control_abstention_integrity"] == 1.0
    assert report["checks"]["negative_control_abstention_all_profiles"] is True
    assert report["metrics"]["min_partial_evidence_abstention_integrity"] == 1.0
    assert report["checks"]["partial_evidence_abstention_all_profiles"] is True
    assert report["metrics"]["min_contrastive_control_accuracy"] == 1.0
    assert report["checks"]["contrastive_control_accuracy_all_profiles"] is True
    assert report["metrics"]["min_dense_embedding_ann_cost_advantage_proxy"] >= 2.0
    assert report["metrics"]["real_pretrained_embedding_reference_profile_count"] == 0.0
    assert report["metrics"]["real_pretrained_embedding_faiss_reference_profile_count"] == 0.0
    assert report["metrics"]["real_cross_encoder_reference_profile_count"] == 0.0
    assert report["checks"]["dense_embedding_cost_advantage_all_profiles"] is True
    assert report["metrics"]["min_sparse_diffusion_real_data_denoise_accuracy"] == 1.0
    assert report["metrics"]["min_sparse_diffusion_real_data_event_cost_advantage"] >= 2.0
    assert report["metrics"]["min_sparse_diffusion_real_data_partition_integrity"] == 1.0
    assert report["metrics"]["min_sparse_diffusion_real_data_single_pass_integrity"] == 1.0
    assert report["checks"]["sparse_diffusion_real_data_denoise_all_profiles"] is True
    assert report["checks"]["sparse_diffusion_real_data_cost_advantage_all_profiles"] is True

    summary = module.format_real_data_external_validity_ladder_summary(report)
    assert "Real Data External Validity Ladder Summary" in summary
    assert "min_negative_control_cost_advantage_proxy" in summary
    assert "min_partial_evidence_cost_advantage_proxy" in summary
    assert "min_contrastive_control_cost_advantage_proxy" in summary
    assert "min_dense_embedding_ann_cost_advantage_proxy" in summary
    assert "real_pretrained_embedding_reference_profile_count" in summary
    assert "real_pretrained_embedding_faiss_reference_profile_count" in summary
    assert "real_cross_encoder_reference_profile_count" in summary
    assert "min_sparse_diffusion_real_data_event_cost_advantage" in summary
    assert "tiny" in summary
    assert "pilot" in summary
