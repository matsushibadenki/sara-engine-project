# パス: src/sara_engine/utils/snn_visualizer.py
# タイトル: SNNデバッグ・可視化ツール
# 目的: スパイクのラスタープロット、アテンションヒートマップ、および膜電位の統計分布ヒストグラムを生成し、SNN内部の挙動を詳細に分析する。

import os

class SNNVisualizer:
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def generate_ascii_raster_plot(self, spike_history: list[list[int]], filename: str = "raster_plot.txt"):
        """
        ターミナルやテキストで確認できる軽量なラスタープロット（省エネ・非依存実装）
        縦軸：ニューロン、横軸：時間
        """
        if not spike_history:
            return

        num_neurons = len(spike_history[0])
        time_steps = len(spike_history)
        
        filepath = os.path.join(self.workspace_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== Spike Raster Plot ===\n")
            f.write("Time: " + "".join(f"{t%10}" for t in range(time_steps)) + "\n")
            f.write("-" * (time_steps + 6) + "\n")
            
            for n in range(num_neurons):
                line = f"N{n:03d}| "
                for t in range(time_steps):
                    if spike_history[t][n] == 1:
                        line += "|"
                    else:
                        line += " "
                f.write(line + "\n")
                
        print(f"ラスタープロットを保存しました: {filepath}")

    def generate_ascii_attention_heatmap(self, attention_history: list[list[float]], filename: str = "attention_heatmap.txt"):
        """
        時間経過に伴うアテンションスコアの推移をテキストでヒートマップ化
        縦軸：現在時刻、横軸：過去の時刻（参照先）
        """
        filepath = os.path.join(self.workspace_dir, filename)
        
        # スコアを視覚的な文字にマッピング
        def get_char_for_score(score: float) -> str:
            if score == 0: return "."
            elif score < 1.0: return "-"
            elif score < 2.0: return "+"
            elif score < 3.0: return "*"
            else: return "#"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== Attention Heatmap (Overlap) ===\n")
            f.write("Y-axis: Current Time, X-axis: Target Past Time\n")
            f.write("Legend: [.]=0, [-]<1, [+]<2, [*]<3, [#]>=3\n\n")
            
            for current_t, scores in enumerate(attention_history):
                line = f"T{current_t:03d}| "
                for past_t_score in scores:
                    line += get_char_for_score(past_t_score)
                f.write(line + "\n")
                
        print(f"アテンションヒートマップを保存しました: {filepath}")

    def generate_ascii_membrane_potential_histogram(self, potentials: list[float], threshold: float, bins: int = 10, filename: str = "membrane_potential_hist.txt"):
        """
        膜電位（または内部スコア）の分布をヒストグラムとして出力する。
        発火閾値に対する分布の偏りを診断し、発火の暴発や不発を検出する。
        """
        filepath = os.path.join(self.workspace_dir, filename)
        
        if not potentials:
            return

        min_val = min(potentials)
        max_val = max(potentials)
        
        # 値の範囲が極端に狭い場合の安全策
        if max_val == min_val:
            max_val = min_val + 1.0

        bin_width = (max_val - min_val) / bins
        histogram = [0] * bins

        # データを各ビンに振り分ける
        for p in potentials:
            # ビンインデックスの計算（行列演算を使わずに算出）
            bin_idx = int((p - min_val) / bin_width)
            if bin_idx >= bins:
                bin_idx = bins - 1
            if bin_idx < 0:
                bin_idx = 0
            histogram[bin_idx] += 1

        max_count = max(histogram) if max(histogram) > 0 else 1
        max_bar_length = 40 # バーの最大表示長

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== Membrane Potential Distribution Histogram ===\n")
            f.write(f"Total Samples: {len(potentials)}\n")
            f.write(f"Firing Threshold: {threshold:.2f}\n")
            f.write("-" * 60 + "\n")
            
            for i in range(bins):
                bin_start = min_val + i * bin_width
                bin_end = bin_start + bin_width
                count = histogram[i]
                
                # バーの長さを計算
                bar_length = int((count / max_count) * max_bar_length)
                bar = "#" * bar_length
                
                # 閾値がこのビンの範囲に含まれるか確認してマークをつける
                threshold_mark = "  "
                if bin_start <= threshold <= bin_end:
                    threshold_mark = "<- THRESHOLD"
                
                # 書式を整えて出力
                f.write(f"[{bin_start:5.2f} - {bin_end:5.2f}) | {count:4d} | {bar} {threshold_mark}\n")

        print(f"膜電位分布ヒストグラムを保存しました: {filepath}")