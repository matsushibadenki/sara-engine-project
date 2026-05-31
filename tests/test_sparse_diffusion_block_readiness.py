import importlib.util
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "sparse_diffusion_block_readiness.py")
    spec = importlib.util.spec_from_file_location("sparse_diffusion_block_readiness", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sparse_diffusion_block_readiness"] = module
    spec.loader.exec_module(module)
    return module


def test_sparse_diffusion_block_readiness_passes_default_fixture():
    module = _load_module()

    report = module.build_sparse_diffusion_block_readiness_report()

    assert report["passed"] is True
    assert report["overall_score"] == 1.0
    assert report["metrics"]["sparse_diffusion_partition_integrity"] == 1.0
    assert report["metrics"]["sparse_diffusion_independent_block_integrity"] == 1.0
    assert report["metrics"]["sparse_diffusion_denoise_accuracy"] == 1.0
    assert report["metrics"]["sparse_diffusion_event_cost_advantage"] >= 2.0
    assert report["details"]["policy"]["runtime_backprop_required"] is False
    assert report["details"]["policy"]["gpu_required"] is False


def test_sparse_diffusion_block_readiness_uses_equal_mass_partitions():
    module = _load_module()
    cases = module._fixture_cases()

    partitions = module._partition_cases_by_equal_mass(cases, block_count=3)
    counts = [len(partition) for partition in partitions]

    assert counts == [2, 2, 2]
    assert sorted(case.case_id for partition in partitions for case in partition) == sorted(
        case.case_id for case in cases
    )


def test_sparse_diffusion_block_readiness_rejects_overlap_in_independence_report():
    module = _load_module()
    block_a = module.SparseDiffusionBlock(
        block_id="a",
        uncertainty_min=0.0,
        uncertainty_max=0.5,
        case_ids=["shared"],
        clean_events=set(),
        distractor_events=set(),
    )
    block_b = module.SparseDiffusionBlock(
        block_id="b",
        uncertainty_min=0.5,
        uncertainty_max=1.0,
        case_ids=["shared"],
        clean_events=set(),
        distractor_events=set(),
    )

    report = module._independence_report([block_a, block_b])

    assert report["case_id_overlap_count"] == 1
    assert report["overlapping_case_ids"] == ["shared"]


def test_sparse_diffusion_block_summary_lists_core_checks():
    module = _load_module()
    report = module.build_sparse_diffusion_block_readiness_report()

    summary = module.format_sparse_diffusion_block_summary(report)

    assert "# SARA Sparse Diffusion Block Readiness" in summary
    assert "- partition_integrity: PASS" in summary
    assert "- single_pass_recurrent_integrity: PASS" in summary
