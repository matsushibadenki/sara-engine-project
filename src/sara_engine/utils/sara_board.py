_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/sara_board.py",
    "//": "ファイルの日本語タイトル: Sara-Board (TensorBoard代替)",
    "//": "ファイルの目的や内容: SNNの内部動態（スパイクのラスタープロット、スカラー値）を記録し、外部依存なしで軽量なインタラクティブHTMLレポートを出力する可視化ツール。"
}

import os
import json
from typing import List, Dict, Any

class SaraBoard:
    """
    Biological alternative to TensorBoard.
    Logs spike events and generates a dependency-free interactive HTML dashboard 
    for monitoring network dynamics (Raster plots & firing rates).
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        # mypy対応: 変数の型を明示
        self.spike_history: List[Dict[str, Any]] = []
        self.scalar_history: List[Dict[str, Any]] = []
        
    def log_spikes(self, step: int, layer_name: str, spikes: List[int]) -> None:
        """Log spike occurrences for a specific layer at a given time step."""
        self.spike_history.append({
            "step": step,
            "layer": layer_name,
            "spikes": list(spikes)
        })
        
    def log_scalar(self, step: int, tag: str, value: float) -> None:
        """Log macro metrics like homeostasis threshold or reward."""
        self.scalar_history.append({
            "step": step,
            "tag": tag,
            "value": float(value)
        })
        
    def export_html(self, filename: str = "saraboard.html") -> str:
        """Generate a standalone HTML dashboard using Canvas API."""
        path = os.path.join(self.log_dir, filename)
        spike_data_json = json.dumps(self.spike_history)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sara-Board Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1e1e1e; color: #d4d4d4; padding: 20px; }}
        h1 {{ color: #569cd6; font-weight: normal; margin-bottom: 5px; }}
        p.subtitle {{ color: #888; margin-top: 0; margin-bottom: 20px; }}
        .panel {{ background: #252526; border: 1px solid #3c3c3c; border-radius: 6px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        canvas {{ background: #1e1e1e; border: 1px solid #555; width: 100%; height: 400px; border-radius: 4px; cursor: crosshair; }}
        select {{ background: #3c3c3c; color: #fff; border: 1px solid #555; padding: 5px 10px; border-radius: 3px; font-size: 14px; outline: none; }}
        .controls {{ margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }}
    </style>
</head>
<body>
    <h1>Sara-Board</h1>
    <p class="subtitle">Spike Dynamics & Neuromorphic Monitoring</p>
    
    <div class="panel">
        <div class="controls">
            <label>Target Layer:</label>
            <select id="layerSelect" onchange="drawRaster()"><option value="ALL">ALL Layers</option></select>
        </div>
        <canvas id="rasterCanvas" width="1200" height="400"></canvas>
    </div>

    <script>
        const spikeData = {spike_data_json};
        const rasterCanvas = document.getElementById('rasterCanvas');
        const rCtx = rasterCanvas.getContext('2d');
        const select = document.getElementById('layerSelect');
        
        // Populate layer selection dropdown
        const layers = [...new Set(spikeData.map(d => d.layer))];
        layers.forEach(l => {{
            const opt = document.createElement('option');
            opt.value = l; opt.innerText = l; select.appendChild(opt);
        }});
        
        function drawRaster() {{
            rCtx.clearRect(0, 0, rasterCanvas.width, rasterCanvas.height);
            const selected = select.value;
            const filtered = spikeData.filter(d => selected === 'ALL' || d.layer === selected);
            
            if (filtered.length === 0) return;
            
            const maxStep = Math.max(...filtered.map(d => d.step));
            const maxNeuron = Math.max(1, ...filtered.flatMap(d => d.spikes.length > 0 ? d.spikes : [0]));
            
            const padX = 50, padY = 30;
            const drawW = rasterCanvas.width - padX * 2;
            const drawH = rasterCanvas.height - padY * 2;
            
            // Draw Spikes
            rCtx.fillStyle = '#4EC9B0'; // VSCode theme teal color
            filtered.forEach(d => {{
                const x = padX + (d.step / (maxStep || 1)) * drawW;
                d.spikes.forEach(s => {{
                    const y = rasterCanvas.height - padY - (s / maxNeuron) * drawH;
                    rCtx.fillRect(x, y, 2, 2);
                }});
            }});
            
            // Draw Axes
            rCtx.strokeStyle = '#555';
            rCtx.beginPath();
            rCtx.moveTo(padX, padY); 
            rCtx.lineTo(padX, rasterCanvas.height - padY);
            rCtx.lineTo(rasterCanvas.width - padX, rasterCanvas.height - padY);
            rCtx.stroke();
            
            rCtx.fillStyle = '#888';
            rCtx.font = "12px sans-serif";
            rCtx.fillText("Time (Steps)", rasterCanvas.width / 2, rasterCanvas.height - 5);
            
            rCtx.save();
            rCtx.translate(15, rasterCanvas.height / 2);
            rCtx.rotate(-Math.PI/2);
            rCtx.fillText("Neuron ID", -25, 0);
            rCtx.restore();
        }}
        
        drawRaster();
    </script>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return path