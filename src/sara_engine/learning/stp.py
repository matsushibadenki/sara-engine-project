# {
#     "//": "ディレクトリパス: src/sara_engine/learning/stp.py",
#     "//": "ファイルの日本語タイトル: 短期シナプス可塑性(STP)マネージャー",
#     "//": "ファイルの目的や内容: Tsodyks-Markramモデルによる短期シナプス可塑性を実装。スパイク列に対してミリ秒〜秒スケールでシナプス強度を一時的に変化させ、短期記憶と時間フィルタリング機能を提供。行列演算なしのイベント駆動型。"
# }

import math
from typing import Dict


class ShortTermPlasticityManager:
    """
    Tsodyks-Markram Short Term Plasticity (STP)
    u: 利用率 (facilitation), x: 資源量 (depression)
    """

    def __init__(
        self,
        U: float = 0.2,
        tau_f: float = 600.0,
        tau_d: float = 200.0
    ):
        """
        Args:
            U: 基準放出確率 (Baseline release probability)
            tau_f: 促通の回復時定数 (Facilitation decay time constant)
            tau_d: 抑圧の回復時定数 (Depression recovery time constant)
        """
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d

        # シナプス前ニューロン(pre_id)ごとの状態管理
        self.u: Dict[int, float] = {}
        self.x: Dict[int, float] = {}
        self.last_update: Dict[int, float] = {}

    def reset(self) -> None:
        """状態をリセット（一時記憶の消去）"""
        self.u.clear()
        self.x.clear()
        self.last_update.clear()

    def _recover(self, neuron_id: int, dt: float) -> None:
        """時間経過(dt)による、利用率(u)の減衰と資源量(x)の回復を計算"""
        u_val = self.u.get(neuron_id, self.U)
        x_val = self.x.get(neuron_id, 1.0)

        # 指数減衰・回復
        u_val = self.U + (u_val - self.U) * math.exp(-dt / self.tau_f)
        x_val = 1.0 + (x_val - 1.0) * math.exp(-dt / self.tau_d)

        self.u[neuron_id] = u_val
        self.x[neuron_id] = x_val

    def on_spike(self, neuron_id: int, current_time: float) -> float:
        """
        スパイクが発生した瞬間に状態を更新し、実効的なスケーリング係数を返す。
        w_eff = w * scale
        """
        last_t = self.last_update.get(neuron_id, current_time)
        dt = current_time - last_t

        if dt > 0:
            self._recover(neuron_id, dt)

        u_val = self.u.get(neuron_id, self.U)
        x_val = self.x.get(neuron_id, 1.0)

        # スパイク発生直前の状態を用いて実効スケールを算出
        scale = u_val * x_val

        # スパイクによる状態の更新（Facilitation & Depression）
        u_val = u_val + self.U * (1.0 - u_val)
        x_val = x_val * (1.0 - u_val)

        self.u[neuron_id] = u_val
        self.x[neuron_id] = x_val
        self.last_update[neuron_id] = current_time

        return scale

    def state_dict(self) -> Dict[str, object]:
        return {
            "u": dict(self.u),
            "x": dict(self.x),
            "last_update": dict(self.last_update)
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if "u" in state:
            self.u = {int(k): float(v) for k, v in state["u"].items()}
        if "x" in state:
            self.x = {int(k): float(v) for k, v in state["x"].items()}
        if "last_update" in state:
            self.last_update = {int(k): float(v)
                                for k, v in state["last_update"].items()}
