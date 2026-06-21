import importlib.util
import os
import sys


def _load_fixture_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "research_fixture_readiness.py")
    )
    spec = importlib.util.spec_from_file_location("research_fixture_readiness", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_research_fixtures_pass_readiness():
    module = _load_fixture_module()

    cases = module.load_fixture_cases(module.DEFAULT_FIXTURE_PATH)
    report = module.build_fixture_readiness_report(cases, module.DEFAULT_FIXTURE_PATH)

    assert report["passed"] is True
    assert report["case_count"] >= 8
    assert set(report["task_types"]) >= module.REQUIRED_TASK_TYPES
    assert report["coverage"]["has_noisy_case"] is True
    assert report["coverage"]["has_adversarial_case"] is True
    assert report["coverage"]["has_delayed_recall_case"] is True
    assert report["coverage"]["has_abstention_cases"] is True


def test_fixture_readiness_rejects_missing_required_task_types():
    module = _load_fixture_module()
    cases = [
        {
            "case_id": "qa_only",
            "task_type": "qa",
            "query": "alpha sparse retrieval",
            "document": "alpha sparse retrieval memory document",
            "expected_keywords": ["alpha", "sparse"],
            "expected_behavior": "retrieve",
        }
    ]

    report = module.build_fixture_readiness_report(cases, module.DEFAULT_FIXTURE_PATH)

    assert report["passed"] is False
    assert "delayed" in report["missing_task_types"]
    assert "negative" in report["missing_task_types"]
