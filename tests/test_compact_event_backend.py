from scripts.eval.compact_event_backend import _load
def test_compact_backend_protocol_is_hash_bound():
    protocol=_load();assert protocol["cost_gate"]["independent_process_repetitions"]==3;assert protocol["claim_boundaries"]["stateless_event_path_only"]
