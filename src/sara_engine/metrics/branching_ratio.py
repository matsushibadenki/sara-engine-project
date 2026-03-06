# {
#     "//": "ディレクトリパス: src/sara_engine/metrics/branching_ratio.py",
#     "//": "ファイルの日本語タイトル: 分岐比（Branching Ratio）エスティメーター",
#     "//": "ファイルの目的や内容: ネットワークが「カオスの縁(Edge of Chaos)」にいるかを判定するための指標である分岐比(σ)を測定する。1スパイクが次のステップで平均何スパイクを生むかを計算し、EMAで滑らかに追跡する。"
# }

class BranchingRatioEstimator:
    """
    Spiking Neural Network の分岐比（Branching Ratio: σ）を推定する。
    σ < 1 : 秩序相（発火がすぐ鎮火し、記憶が保持できない）
    σ > 1 : カオス相（発火が爆発し、ノイズで情報が破壊される）
    σ ≈ 1 : 臨界相（Edge of Chaos: 記憶容量と情報伝播が最大化）
    """
    def __init__(self, smoothing_alpha: float = 0.05):
        """
        Args:
            smoothing_alpha: スパイク数の激しい変動を吸収するためのEMA（指数移動平均）の係数。
        """
        self.prev_spikes = 0
        self.ema_sigma = 1.0
        self.alpha = smoothing_alpha

    def update(self, current_spikes: int) -> float:
        """
        現在のステップのスパイク数を受け取り、分岐比σを更新・返却する。
        """
        if self.prev_spikes == 0:
            if current_spikes > 0:
                # 完全に沈黙していた状態から発火が起きた場合は、爆発的な増殖とみなす
                raw_sigma = 2.0
            else:
                # ずっと沈黙している場合は秩序相(減衰)の極致
                raw_sigma = 0.0
        else:
            # σ = 現在の発火数 / 前回の発火数
            raw_sigma = current_spikes / self.prev_spikes

        # 1ステップごとの瞬間的な値で制御するとネットワークが不安定になるため、EMAで平滑化
        self.ema_sigma = (1.0 - self.alpha) * self.ema_sigma + self.alpha * raw_sigma
        
        self.prev_spikes = current_spikes

        return self.ema_sigma