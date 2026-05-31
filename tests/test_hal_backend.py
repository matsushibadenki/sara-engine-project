from sara_engine.core.hal import MockNeuromorphicBackend, PythonBackend


def test_mock_neuromorphic_backend_uses_quantized_weights_and_threshold():
    weights = [
        {0: 0.31, 1: 0.9},
        {1: 0.2, 2: 1.1},
    ]
    backend = MockNeuromorphicBackend()
    backend.set_weights(weights)

    output = backend.propagate([0, 1], threshold=1.0, max_out=2)
    report = backend.mapping_report()

    assert output == [2, 1]
    assert report["weights_mapped"] == 1.0
    assert report["synapse_count"] == 4.0
    assert report["last_event_cost"] == 4.0


def test_mock_neuromorphic_backend_matches_python_order_for_simple_weights():
    weights = [
        {0: 0.75, 1: 0.25},
        {1: 0.75},
    ]
    python_backend = PythonBackend()
    mock_backend = MockNeuromorphicBackend()
    python_backend.set_weights(weights)
    mock_backend.set_weights(weights)

    assert mock_backend.propagate([0, 1], threshold=0.5, max_out=2) == python_backend.propagate(
        [0, 1], threshold=0.5, max_out=2
    )
