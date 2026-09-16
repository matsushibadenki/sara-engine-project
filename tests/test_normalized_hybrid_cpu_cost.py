from scripts.eval.normalized_hybrid_cpu_cost import _load_protocol,_summary
def test_cpu_cost_protocol_is_hash_bound_and_requires_five_processes_per_arm():
    protocol=_load_protocol();assert protocol["repetitions_per_dataset_mode"]==5;assert protocol["independent_process_per_repetition"]
def test_cpu_summary_is_deterministic():
    assert _summary([1.,2.,100.])=={"median":2.,"minimum":1.,"maximum":100.,"median_absolute_deviation":1.}
