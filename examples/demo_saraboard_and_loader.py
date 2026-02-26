_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_saraboard_and_loader.py",
    "//": "ファイルの日本語タイトル: Sara-Board & データローダーのデモ",
    "//": "ファイルの目的や内容: SpikeStreamLoaderで文字ストリームを供給し、学習中のTransformer内部の発火状態をSara-Boardに記録してダッシュボードを出力する。"
}

import os
from sara_engine.utils.sara_board import SaraBoard
from sara_engine.utils.data.dataloader import SpikeStreamLoader
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def char_to_spikes(char: str) -> list:
    """Simple ASCII to Spike ID encoder."""
    return [ord(char)]

def run_demo():
    # Setup paths
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    print("Initializing Sara-Board & SpikeStreamLoader...")
    board = SaraBoard(log_dir=workspace_dir)
    
    # Initialize tiny SNN Transformer for visualization
    config = SNNTransformerConfig(embed_dim=64, num_layers=1)
    model = SpikingTransformerModel(config)
    
    # Setup SpikeStreamLoader
    raw_text = "SARA-Engine: Neuromorphic AI without BP or Matrix Math. " * 3
    loader = SpikeStreamLoader(dataset=list(raw_text), encode_fn=char_to_spikes, time_step=1)
    
    print("Streaming data and logging spikes...")
    # Process stream and log dynamics
    for event in loader.stream():
        step = event["time"]
        token_id = event["spikes"][0]
        
        # Drive the SNN network (STDP Learning)
        model.forward_step(token_id, learning=True, target_id=token_id)
        
        # Log spike activity to Sara-Board
        board.log_spikes(step, layer_name="1. Input Stream", spikes=event["spikes"])
        
        # Get reservoir spikes
        res_spikes = model._get_reservoir_spikes(token_id)
        board.log_spikes(step, layer_name="2. Context Reservoir", spikes=res_spikes)
        
        # Log active readout synapses
        active_readouts = []
        if token_id < len(model.readout_synapses):
            active_readouts = [k for k, w in model.readout_synapses[token_id].items() if w > 0.5]
        if active_readouts:
            board.log_spikes(step, layer_name="3. Readout Activity", spikes=active_readouts)

    # Export dashboard
    html_path = board.export_html("saraboard_demo.html")
    print(f"\nSUCCESS: Processing complete.")
    print(f"Open this file in your browser to view the Raster Plot: file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    run_demo()