"""Long-horizon Event Memory retention stays bounded and source-pinned."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval/structural_event_memory_long_horizon.py"
SPEC = importlib.util.spec_from_file_location("structural_event_memory_long_horizon", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_folded_signature_is_deterministic_and_bounded():
    hashes = [f"{index:064x}" for index in range(101)]
    first = MODULE.folded_signature(hashes, 64)
    assert first == MODULE.folded_signature(tuple(reversed(hashes)), 64)
    assert 1 <= len(first) <= 64
    assert len(first) == len(set(first))


def test_frozen_long_horizon_profiles_pass():
    protocol = json.loads(MODULE.PROTOCOL.read_text())
    assert MODULE.digest(MODULE.PROTOCOL) == MODULE.PROTOCOL_SHA256
    manifest = ROOT / protocol["manifest"]
    assert MODULE.digest(manifest) == protocol["manifest_sha256"]
    report = MODULE.build_report(MODULE.load_rows(manifest), protocol)
    assert report["passed"] is True
    assert report["domain_counts"] == {"docs.python.org": 101, "www.rfc-editor.org": 101}
    assert report["profiles"]["linear"]["retrieval_recall"] == 1.0
    assert report["profiles"]["logarithmic"]["retrieval_recall"] == 1.0
    assert report["profiles"]["linear"]["cache_entry_count"] <= 8
    assert report["profiles"]["logarithmic"]["cache_entry_count"] <= 8
    assert {
        name: (values["retrieval_recall"], values["total_event_cost"])
        for name, values in report["profiles"].items()
    } == {
        "fixed": (1.0, 64),
        "linear": (1.0, 64),
        "logarithmic": (1.0, 64),
    }
    assert report["candidate_scored"] is False
