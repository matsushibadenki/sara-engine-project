from sara_engine.models.spiking_v_jepa import SpikingVJEPA

def generate_video_stream(num_frames=10, dim=64):
    """Generates a simple moving pattern in a spike stream."""
    stream = []
    for t in range(num_frames):
        # A simple pattern that moves by 1 bit each frame
        frame = [0] * dim
        pos = (t * 4) % dim
        for i in range(4):
            frame[(pos + i) % dim] = 1
        stream.append([i for i, s in enumerate(frame) if s == 1])
    return stream

def test_v_jepa_temporal_prediction():
    print("Testing Spiking V-JEPA Temporal Prediction...")
    
    input_dim = 64
    embed_dim = 32
    hidden_dim = 64
    
    model = SpikingVJEPA(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        learning_rate=0.1
    )
    
    video = generate_video_stream(num_frames=20)
    
    # Initial error
    initial_error = model.step(video, learning=False)
    print(f"Initial Avg Error: {initial_error:.4f}")
    
    # Training
    for epoch in range(30):
        error = model.step(video, learning=True)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Error: {error:.4f}")
            
    final_error = model.step(video, learning=False)
    print(f"Final Avg Error: {final_error:.4f}")
    
    assert final_error <= initial_error, f"Error did not decrease: {initial_error} -> {final_error}"
    print("Test Passed: Spatiotemporal error decreased.")

if __name__ == "__main__":
    try:
        test_v_jepa_temporal_prediction()
    except Exception as e:
        print(f"Test Failed: {e}")
        exit(1)
