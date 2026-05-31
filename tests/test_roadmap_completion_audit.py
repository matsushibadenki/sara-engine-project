import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "eval" / "roadmap_completion_audit.py"
    spec = importlib.util.spec_from_file_location("roadmap_completion_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_roadmap_completion_audit_passes_for_current_roadmap():
    module = _load_module()
    report = module.audit_roadmap_path(ROOT / "doc" / "ROADMAP.md")

    assert report["passed"] is True
    assert report["status"] == "complete"
    assert report["closure_present"] is True
    assert report["closure_done_count"] >= 5
    assert report["missing_markers"] == []
    assert report["unchecked_marker_count"] == 0
    assert report["long_term_backlog_line_count"] >= 1
    assert report["candidate_line_count"] >= 1


def test_roadmap_completion_audit_detects_missing_closure_marker():
    module = _load_module()
    report = module.audit_roadmap_text(
        "ROADMAP closure audit\n"
        "* DONE: release-critical path is complete.\n"
        "* DONE: observed-only items are categorized.\n"
        "* DONE: roadmap completion audit is present.\n"
    )

    assert report["passed"] is False
    assert "long-term research backlog" in report["missing_markers"]
    assert "research product completion gate" in report["missing_markers"]


def test_roadmap_completion_audit_ignores_markers_outside_done_closure_section():
    module = _load_module()
    report = module.audit_roadmap_text(
        "release-critical path observed-only long-term research backlog roadmap completion audit\n"
        "* **ROADMAP closure audit:**\n"
        "  * TODO: release-critical path still needs review.\n"
        "* **Next section:**\n"
        "  * DONE: observed-only text outside closure should not satisfy the audit.\n"
    )

    assert report["passed"] is False
    assert report["closure_present"] is True
    assert report["closure_done_count"] == 0
    assert "release-critical path" in report["missing_markers"]
