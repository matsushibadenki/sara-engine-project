import importlib.util
import json
import os


def _load_module():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase9_completion_gate.py")
    )
    spec = importlib.util.spec_from_file_location("phase9_completion_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase9_gate_accepts_complete_managed_package(tmp_path):
    module = _load_module()
    module.PROJECT_ROOT = str(tmp_path)
    output = tmp_path / "workspace" / "artifact.json"
    output.parent.mkdir()
    output.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "sara-research-benchmark-manifest-v1",
                "dry_run": False,
                "commands": [
                    {
                        "command_id": "fixture",
                        "status": "passed",
                        "returncode": 0,
                        "managed_outputs": [str(output)],
                    }
                ],
                "what_is_proven": ["fixture"],
                "what_is_not_proven": ["physical energy"],
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Recommended Command\nWhat Is Proven\nWhat Is Not Proven\nOutput Policy", encoding="utf-8")
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        "\n".join(
            json.dumps({"task_type": case})
            for case in ["qa", "negative", "partial", "contrastive", "noisy", "adversarial", "delayed"]
        ),
        encoding="utf-8",
    )
    report = module.build_report(manifest_path=str(manifest), protocol_path=str(protocol), fixture_path=str(fixture))
    assert report["phase9_complete"] is True


def test_phase9_gate_rejects_dry_run_and_missing_outputs(tmp_path):
    module = _load_module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "sara-research-benchmark-manifest-v1",
                "dry_run": True,
                "commands": [{"command_id": "x", "status": "planned", "returncode": None, "managed_outputs": [str(tmp_path / "x")]}],
                "what_is_proven": ["x"],
                "what_is_not_proven": ["y"],
            }
        ),
        encoding="utf-8",
    )
    assert module._check_manifest(str(manifest))["passed"] is False
