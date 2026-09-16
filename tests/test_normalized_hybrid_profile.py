from scripts.eval.normalized_hybrid_profile import _load_protocol
def test_component_profile_protocol_is_hash_bound():
    protocol=_load_protocol();assert protocol["runs_per_dataset_mode"]==1;assert "neuron step" in protocol["scopes"]
