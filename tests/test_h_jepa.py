from sara_engine.models.spiking_jepa import SpikingJEPA

def test_h_jepa_convergence():
    print("Testing Hierarchical JEPA Convergence...")
    
    # Simple hierarchy: Layer 1 (64 -> 32), Layer 2 (32 -> 16)
    layer_configs = [
        {"input_dim": 64, "embed_dim": 32},
        {"input_dim": 32, "embed_dim": 16}
    ]
    
    model = SpikingJEPA(layer_configs=layer_configs, learning_rate=0.1)
    
    # Fixed pattern for testing
    input_pattern = [1 if i % 4 == 0 else 0 for i in range(64)]
    target_pattern = [1 if i % 4 == 0 else 0 for i in range(64)]
    
    input_spikes = [i for i, s in enumerate(input_pattern) if s == 1]
    target_spikes = [i for i, s in enumerate(target_pattern) if s == 1]
    
    initial_surprise = 0.0
    final_surprise = 0.0
    
    # Initial forward
    _, initial_surprise = model.forward(input_spikes, target_spikes, learning=False)
    print(f"Initial Surprise: {initial_surprise}")
    
    # Training loop
    for epoch in range(50):
        _, surprise = model.forward(input_spikes, target_spikes, learning=True)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Surprise: {surprise}")
            
    _, final_surprise = model.forward(input_spikes, target_spikes, learning=False)
    print(f"Final Surprise: {final_surprise}")
    
    # surprise signal is (accuracy * 2) - 1. -1 is bad, 1 is good.
    # So we want it to increase towards 1.
    assert final_surprise >= initial_surprise, f"Surprise signal did not improve: {initial_surprise} -> {final_surprise}"
    print("Test Passed: Surprise signal improved/stabilized.")

if __name__ == "__main__":
    try:
        test_h_jepa_convergence()
    except Exception as e:
        print(f"Test Failed: {e}")
        exit(1)
