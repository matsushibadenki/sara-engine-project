import importlib.util
import json
from pathlib import Path

from sara_engine.memory.sdr import SDREncoder


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chat_encoder_comparison", ROOT / "scripts/eval/chat_encoder_compatibility.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_comparison_uses_real_encoder_and_isolated_vocabulary():
    first = module.IsolatedChatEncoder()
    second = module.IsolatedChatEncoder()
    assert isinstance(first.agent.encoder, SDREncoder)
    assert first("north door state") == second("north door state")
    before = dict(second.agent.encoder.tokenizer.vocab)
    first("unique additional vocabulary")
    assert second.agent.encoder.tokenizer.vocab == before


def test_encoder_contract_reports_failures_without_promoting_chat():
    protocol = json.loads((ROOT / "data/processed/benchmark_fixtures/chat_encoder_compatibility.json").read_text())
    report = module.evaluate(protocol)
    assert report["case_count"] == 18
    assert report["normal_chat_promotion"] is False
    assert any(row["encoder_max_signature_width"] > 64 for row in report["cases"])
    assert report["all_cases_passed"] == all(row["verified_correct"] for row in report["cases"])
    assert all(row["candidate_build_and_query_cpu_ms"] >= 0 for row in report["cases"])
