import os
from typing import Any

np: Any
try:
    import numpy as np
except ImportError:
    np = None
from sara_engine.utils.data.event_parser import DVSEventParser
from sara_engine.utils.data.event_transform import EventDownsampler

def test_dvs_pipeline():
    print("Testing DVS Data Pipeline...")
    
    # Create a synthetic CSV event file
    csv_content = "t,x,y,p\n0.1,10,10,1\n0.15,12,12,0\n0.2,10,10,1\n0.5,50,50,1\n"
    csv_path = "tests/test_events.csv"
    os.makedirs("tests", exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(csv_content)
        
    parser = DVSEventParser()
    events = parser.parse_csv_events(csv_path)
    print(f"Parsed {len(events)} events.")
    assert len(events) == 4
    
    downsampler = EventDownsampler()
    downsampled = downsampler.spatial_downsample(events, factor=2)
    print(f"Downsampled to {len(downsampled)} events.")
    assert downsampled[0][1] == 5 # 10 // 2

    temporal = downsampler.temporal_downsample(downsampled, time_step=0.2)
    assert temporal[0][0] == 0.0
    
    spike_trains = parser.to_spike_trains(downsampled, width=32, height=32)
    print(f"Generated spike trains for {len(spike_trains)} bins.")

    npz_path = "tests/test_events.npz"
    if np is not None:
        np.savez(
            npz_path,
            t=np.array([0.1, 0.2], dtype=np.float32),
            x=np.array([1, 2], dtype=np.int32),
            y=np.array([3, 4], dtype=np.int32),
            p=np.array([1, 0], dtype=np.int32),
        )
        npz_events = parser.parse_npz_events(npz_path)
        assert len(npz_events) == 2
    else:
        print("NumPy not available, skipping NPZ parse test.")
    
    os.remove(csv_path)
    if os.path.exists(npz_path):
        os.remove(npz_path)
    print("DVS Pipeline Test Passed.")

if __name__ == "__main__":
    try:
        test_dvs_pipeline()
    except Exception as e:
        print(f"Test Failed: {e}")
        exit(1)
