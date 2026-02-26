_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/edge/exporter.py",
    "//": "ファイルの日本語タイトル: エッジ用モデルエクスポーター",
    "//": "ファイルの目的や内容: 学習済みのSARAモデルからシナプス重みだけを抽出し、エッジデバイスで読み込める軽量なフォーマットにシリアライズする。"
}

import json
from typing import Any

def export_for_edge(model: Any, filepath: str):
    """
    Extracts essential inference parameters (e.g., readout synapses, context length)
    from a SpikingTransformerModel and saves it as a lightweight JSON for Sara-Edge.
    """
    edge_data = {
        "context_length": getattr(model, "context_length", 64),
        "embed_dim": model.config.embed_dim if hasattr(model, "config") else 64,
        "total_readout_size": getattr(model, "total_readout_size", 8192 + 64),
        "readout_synapses": []
    }
    
    # Convert readout_synapses (List[Dict[int, float]]) to a JSON-serializable list
    if hasattr(model, "readout_synapses"):
        for synapses in model.readout_synapses:
            edge_data["readout_synapses"].append(synapses)
            
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(edge_data, f)
    
    print(f"Model successfully exported for Sara-Edge at: {filepath}")