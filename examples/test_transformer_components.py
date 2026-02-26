_FILE_INFO = {
    "//": "ディレクトリパス: examples/test_transformer_components.py",
    "//": "ファイルの日本語タイトル: Transformerコンポーネントテスト",
    "//": "ファイルの目的や内容: 新規作成したSpikeDropoutとSpikeLayerNormの挙動を検証し、ログをworkspaceディレクトリに出力する。"
}

import os
import random
from sara_engine.nn.dropout import SpikeDropout
from sara_engine.nn.normalization import SpikeLayerNorm

def run_test():
    # Create workspace directory for logs
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "transformer_components_test.log")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== SNN Transformer Components Test Log ===\n")
        
        # Generate dummy input spikes
        input_spikes = random.sample(range(1000), 100)
        f.write(f"Original spikes count: {len(input_spikes)}\n")
        
        # Test SpikeDropout
        dropout = SpikeDropout(p=0.2)
        dropped_spikes = dropout(input_spikes, learning=True)
        f.write(f"After SpikeDropout (p=0.2): {len(dropped_spikes)}\n")
        
        # Test SpikeLayerNorm (Homeostasis)
        norm = SpikeLayerNorm(target_spikes=30)
        norm_spikes = norm(dropped_spikes, learning=True)
        f.write(f"After SpikeLayerNorm (target=30): {len(norm_spikes)}\n")
        
        # Test Low Activity Handling in SpikeLayerNorm
        low_spikes = random.sample(range(1000), 5)
        f.write(f"Low activity original count: {len(low_spikes)}\n")
        boosted_spikes = norm(low_spikes, learning=True)
        f.write(f"After SpikeLayerNorm (Boosted): {len(boosted_spikes)}\n")

        f.write("Test completed successfully.\n")

    print(f"Log generated at: {log_file_path}")
    print("Note: If the latest code changes are not reflected, please ensure you ran 'pip install -e .'")

if __name__ == "__main__":
    run_test()