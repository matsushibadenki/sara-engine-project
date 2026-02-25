_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/sara_board.py",
    "//": "ファイルの日本語タイトル: Sara-Board ロガーおよび可視化ツール",
    "//": "ファイルの目的や内容: TensorBoardの代替。SNN固有のスパイク発火やSTDPによるシナプス荷重変化を軽量に記録し、外部ライブラリ（NumPy等）に依存せず純粋なPythonのみでHTML/SVGダッシュボードを生成する。"
}

import json
import os
from typing import Dict, List, Any

class SaraBoardLogger:
    """
    SNNの内部状態やダイナミクスを時系列で記録するロガー。
    重いプロット処理や行列演算を避け、軽量なJSON Lines形式でイベントストリームを永続化する。
    """
    def __init__(self, log_dir: str = "workspace/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.spike_log_path = os.path.join(self.log_dir, "spikes.jsonl")
        self.weight_log_path = os.path.join(self.log_dir, "weights.jsonl")
        
        # 初期化時に過去のログファイルをクリア
        open(self.spike_log_path, 'w', encoding='utf-8').close()
        open(self.weight_log_path, 'w', encoding='utf-8').close()

    def log_spike(self, timestamp: int, layer_name: str, neuron_id: int) -> None:
        """
        発火イベント（スパイク）を記録。のちにラスタープロットの生成に利用する。
        """
        record = {
            "t": timestamp,
            "layer": layer_name,
            "nid": neuron_id
        }
        with open(self.spike_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

    def log_weight_change(self, timestamp: int, pre_id: int, post_id: int, new_weight: float) -> None:
        """
        STDPや局所学習則によるシナプス荷重の変化を記録。
        トポロジーの動的変化の監視に利用する。
        """
        record = {
            "t": timestamp,
            "pre": pre_id,
            "post": post_id,
            "w": round(new_weight, 4)
        }
        with open(self.weight_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

    def get_spike_history(self) -> List[Dict[str, Any]]:
        """
        記録された発火履歴をオンデマンドで取得する。
        """
        history = []
        if os.path.exists(self.spike_log_path):
            with open(self.spike_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
        return history


class SaraBoardVisualizer:
    """
    記録されたSNNのログデータから、依存関係なし（NumPy/Matplotlib不要）で
    SVGを含むHTMLダッシュボードを生成するツール。
    """
    def __init__(self, log_dir: str = "workspace/logs"):
        self.log_dir = log_dir
        self.spike_log_path = os.path.join(self.log_dir, "spikes.jsonl")
        self.weight_log_path = os.path.join(self.log_dir, "weights.jsonl")

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data

    def generate_dashboard(self, output_html: str = "workspace/sara_dashboard.html") -> str:
        """HTMLファイルを生成し、出力先のパスを返す"""
        spikes = self._read_jsonl(self.spike_log_path)
        weights = self._read_jsonl(self.weight_log_path)

        html = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='UTF-8'><title>Sara-Board Dashboard</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; padding: 20px; }",
            "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
            "h2 { color: #2980b9; margin-top: 30px; }",
            ".card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }",
            "</style></head><body>",
            "<h1>SARA Engine Dashboard (SNN Dynamics)</h1>"
        ]

        # 1. Raster Plot (Spikes)
        if spikes:
            html.append("<div class='card'><h2>Spike Raster Plot</h2>")
            width, height = 800, 300
            max_t = max(s['t'] for s in spikes)
            min_t = min(s['t'] for s in spikes)
            nids = [s['nid'] for s in spikes]
            max_nid = max(nids) if nids else 1
            min_nid = min(nids) if nids else 0
            
            html.append(f"<svg width='{width}' height='{height}' style='background: #fff; border: 1px solid #ddd; border-radius: 4px;'>")
            for s in spikes:
                # 時間 t を x座標 にマッピング
                x = 20 + (s['t'] - min_t) / (max_t - min_t + 1e-5) * (width - 40)
                # ニューロンID を y座標 にマッピング (IDが大きいほど上に描画)
                y_range = max_nid - min_nid + 1e-5
                y = height - 20 - ((s['nid'] - min_nid) / y_range) * (height - 40)
                html.append(f"<circle cx='{x}' cy='{y}' r='4' fill='#3498db' opacity='0.7' />")
            html.append("</svg></div>")

        # 2. Weight Evolution (Synapses)
        if weights:
            html.append("<div class='card'><h2>Synaptic Weight Evolution (STDP)</h2>")
            width, height = 800, 250
            max_t = max(w['t'] for w in weights)
            min_t = min(w['t'] for w in weights)
            max_w = max(w['w'] for w in weights)
            min_w = min(w['w'] for w in weights)
            
            if max_w == min_w:
                max_w += 0.1
                min_w -= 0.1

            html.append(f"<svg width='{width}' height='{height}' style='background: #fff; border: 1px solid #ddd; border-radius: 4px;'>")
            path_d = ""
            for i, w in enumerate(weights):
                x = 20 + (w['t'] - min_t) / (max_t - min_t + 1e-5) * (width - 40)
                y = height - 20 - ((w['w'] - min_w) / (max_w - min_w)) * (height - 40)
                
                if i == 0:
                    path_d += f"M {x} {y} "
                else:
                    path_d += f"L {x} {y} "
                    
            # グラフの線
            html.append(f"<path d='{path_d}' fill='none' stroke='#e74c3c' stroke-width='2' />")
            # データポイント
            for w in weights:
                x = 20 + (w['t'] - min_t) / (max_t - min_t + 1e-5) * (width - 40)
                y = height - 20 - ((w['w'] - min_w) / (max_w - min_w)) * (height - 40)
                html.append(f"<circle cx='{x}' cy='{y}' r='3' fill='#c0392b' />")
            
            html.append("</svg></div>")

        if not spikes and not weights:
            html.append("<p>No data recorded in the log files.</p>")

        html.append("</body></html>")
        
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
            
        return output_html