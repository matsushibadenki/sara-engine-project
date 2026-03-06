# {
#     "//": "ディレクトリパス: src/sara_engine/learning/homeostatic_plasticity.py",
#     "//": "ファイルの日本語タイトル: ホメオスタシス可塑性マネージャー",
#     "//": "ファイルの目的や内容: 内因性可塑性（Intrinsic Plasticity）を実装。各ニューロンの発火率を監視し、目標値より高い場合はしきい値を上げ、低い場合は下げることで、ネットワークを自己組織化臨界状態(SOC)へ導く。"
# }

from typing import List, Dict

class HomeostaticPlasticityManager:
    """
    発火率を一定に保つためのしきい値(Threshold)制御。
    θ ← θ + η * (実際の活動 - 目標活動)
    """

    def __init__(
        self,
        n_neurons: int,
        target_rate: float = 0.05,
        learning_rate: float = 0.01,
        min_threshold: float = -50.0,
        max_threshold: float = 0.0
    ):
        """
        Args:
            n_neurons: ニューロン数
            target_rate: 1ステップあたりの目標発火確率 (例: 0.05 = 20ステップに1回発火)
            learning_rate: しきい値更新の速さ
        """
        self.n = n_neurons
        self.target_rate = target_rate
        self.eta = learning_rate
        self.min_theta = min_threshold
        self.max_theta = max_threshold

        # 各ニューロンの移動平均発火率 (0.0 ~ 1.0)
        self.firing_rates = [target_rate for _ in range(n_neurons)]
        # 指数移動平均の平滑化係数
        self.alpha = 0.01

    def update(self, fired_ids: List[int], current_thresholds: List[float]) -> List[float]:
        """
        現在の発火状況に基づき、新しいしきい値リストを計算して返す。
        """
        new_thresholds = list(current_thresholds)
        fired_set = set(fired_ids)

        for i in range(self.n):
            # 1. 実際の活動(スパイク有無)を指数移動平均で更新
            activity = 1.0 if i in fired_set else 0.0
            self.firing_rates[i] = (1.0 - self.alpha) * self.firing_rates[i] + self.alpha * activity

            # 2. しきい値の調整: 発火しすぎなら上げ(難しく)、発火不足なら下げ(易しく)
            # ※Izhikevichモデルではしきい値は通常30mV固定だが、
            # 生物的にはベースラインの膜電位やしきい値ポテンシャルが変化する。
            # ここでは「発火条件」としての相対的なしきい値オフセットとして扱う。
            error = self.firing_rates[i] - self.target_rate
            new_thresholds[i] += self.eta * error

            # 安全のための制限
            new_thresholds[i] = max(self.min_theta, min(self.max_theta, new_thresholds[i]))

        return new_thresholds

    def state_dict(self) -> Dict[str, object]:
        return {
            "firing_rates": list(self.firing_rates),
            "target_rate": self.target_rate
        }

    def load_state_dict(self, state: Dict[str, object]):
        if "firing_rates" in state:
            self.firing_rates = [float(x) for x in state["firing_rates"]] # type: ignore
        if "target_rate" in state:
            self.target_rate = float(state["target_rate"]) # type: ignore