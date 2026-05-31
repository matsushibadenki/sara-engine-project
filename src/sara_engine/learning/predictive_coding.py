# Directory Path: src/sara_engine/learning/predictive_coding.py
# English Title: Precision-Weighted Predictive Coding and Target Propagation
# Purpose/Content: 誤差逆伝播法に依存しない局所学習。予測誤差の分散(Precision)に基づく動的な学習率の調整によりノイズへのロバスト性を向上。また、DTPのLTD演算をスパース化し省エネ化を実現。他のモジュールやテストとの後方互換性を完全維持。

from typing import List, Dict, Set, Optional

_WEIGHT_CAP: float = 3.0
_ERROR_EMA_ALPHA: float = 0.1

class PredictiveCodingManager:
    """精度重み付け(Precision-Weighted)予測符号化に基づく局所的シナプス可塑性マネージャー"""

    def __init__(self, learning_rate: float = 0.05) -> None:
        # 他モジュールとの後方互換性を維持するため `learning_rate` という名前で保持します
        self.learning_rate = learning_rate
        self.prediction_error_ema: float = 1.0  
        self.error_variance_ema: float = 0.5 # 予測誤差の分散を追跡
        self.total_updates: int = 0

    def get_status(self, lang: str = "en") -> str:
        precision = self.get_precision()
        messages = {
            "en": f"PredictiveCoding: Error EMA={self.prediction_error_ema:.3f}, Precision={precision:.3f}",
            "ja": f"PredictiveCoding: 誤差EMA={self.prediction_error_ema:.3f}, 予測精度(Precision)={precision:.3f}",
            "fr": f"PredictiveCoding: Erreur EMA={self.prediction_error_ema:.3f}, Précision={precision:.3f}"
        }
        return messages.get(lang, messages["en"])

    def reset(self) -> None:
        self.prediction_error_ema = 1.0
        self.error_variance_ema = 0.5
        self.total_updates = 0

    def get_prediction_accuracy(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.prediction_error_ema))
        
    def get_precision(self) -> float:
        """ノイズの少なさ（精度）を計算。分散が小さいほど精度が高い。"""
        return 1.0 / (self.error_variance_ema + 1e-4)

    def record_error(self, error_rate: float) -> float:
        """予測誤差率と分散を追跡し、現在のPrecisionに基づく動的学習率を返す"""
        clamped = max(0.0, min(1.0, error_rate))
        
        # 誤差の分散(Variance)を更新
        error_diff = clamped - self.prediction_error_ema
        self.error_variance_ema = (1.0 - _ERROR_EMA_ALPHA) * self.error_variance_ema + _ERROR_EMA_ALPHA * (error_diff ** 2)
        
        self.prediction_error_ema = (1.0 - _ERROR_EMA_ALPHA) * self.prediction_error_ema + _ERROR_EMA_ALPHA * clamped
        self.total_updates += 1
        
        # Precision-weighted Learning Rate: 予測が安定している(分散小)時ほど、起きた誤差から強く学ぶ
        dynamic_lr = self.learning_rate * min(5.0, self.get_precision())
        return dynamic_lr

    @staticmethod
    def update_forward(
        forward_weights: List[Dict[int, float]],
        in_spikes: List[int],
        state_spikes: List[int],
        error_spikes: List[int],
        lr: float = 0.05,
    ) -> None:
        # 省エネ化: 誤差が全くない場合は計算自体をスキップ
        if not error_spikes: return
        
        state_set = set(state_spikes)
        for s in error_spikes:
            if s < len(forward_weights):
                for t in state_set:
                    forward_weights[s][t] = min(_WEIGHT_CAP, forward_weights[s].get(t, 0.0) + lr)

    @staticmethod
    def update_backward(
        backward_weights: List[Dict[int, float]],
        prev_state_spikes: List[int],
        current_in_spikes: List[int],
        predicted_in_spikes: Set[int],
        lr: float = 0.1,
    ) -> None:
        in_set = set(current_in_spikes)
        false_positives = predicted_in_spikes - in_set

        for s in prev_state_spikes:
            if s < len(backward_weights):
                for t in in_set:
                    backward_weights[s][t] = min(_WEIGHT_CAP, backward_weights[s].get(t, 0.0) + lr)

                # 予測したのに来なかった入力への結合の減衰 (必要な対象のみに絞る)
                if false_positives:
                    for t in false_positives:
                        if t in backward_weights[s]:
                            backward_weights[s][t] -= lr * 0.5
                            if backward_weights[s][t] <= 0.01: # 微小結合の厳密なプルーニング
                                del backward_weights[s][t]


class TargetPropagationManager:
    """ターゲット伝播(Target Propagation)とスパースDTPマネージャー"""

    def __init__(self, lr: float = 0.1, inverse_lr: float = 0.05) -> None:
        self.lr = lr
        self.inverse_lr = inverse_lr
        self.inverse_weights: Dict[int, Dict[int, float]] = {}

    def get_status(self, lang: str = "en") -> str:
        messages = {
            "en": f"TargetPropManager: Inverse connections ready. LR={self.lr}",
            "ja": f"TargetPropManager: 逆写像接続準備完了。学習率={self.lr}",
            "fr": f"TargetPropManager: Connexions inverses prêtes. LR={self.lr}"
        }
        return messages.get(lang, messages["en"])

    def reset(self) -> None:
        self.inverse_weights.clear()

    @staticmethod
    def apply_target(
        weights: List[Dict[int, float]],
        in_spikes: List[int],
        out_spikes: List[int],
        target_spikes: List[int],
        lr: float = 0.1,
    ) -> None:
        target_set = set(target_spikes)
        out_set = set(out_spikes)
        false_fires = out_set - target_set

        for s in in_spikes:
            if s < len(weights):
                for t in target_set:
                    weights[s][t] = min(_WEIGHT_CAP, weights[s].get(t, 0.0) + lr)

                # スパース演算化: 発火すべきでなかったニューロンのみを減衰対象とする
                if false_fires:
                    for t in false_fires:
                        if t in weights[s]:
                            weights[s][t] -= lr
                            if weights[s][t] <= 0.01:
                                del weights[s][t]

    def update_inverse(
        self,
        out_spikes: List[int],
        in_spikes: List[int],
        lr: Optional[float] = None,
    ) -> None:
        actual_lr = lr if lr is not None else self.inverse_lr
        in_set = set(in_spikes)

        for o in out_spikes:
            if o not in self.inverse_weights:
                self.inverse_weights[o] = {}
            w_map = self.inverse_weights[o]

            for i in in_set:
                w_map[i] = min(_WEIGHT_CAP, w_map.get(i, 0.0) + actual_lr)

            # 既存のキーをリスト化して安全に反復・削除
            keys_to_check = list(w_map.keys())
            for i in keys_to_check:
                if i not in in_set:
                    w_map[i] -= actual_lr * 0.3
                    if w_map[i] <= 0.05: # 計算リソース節約のため早めに刈り込み
                        del w_map[i]

    def compute_local_target(
        self,
        upper_target_spikes: List[int],
        current_out_spikes: List[int],
        current_in_spikes: List[int],
        threshold: float = 0.5,
    ) -> List[int]:
        g_target: Dict[int, float] = {}
        for o in upper_target_spikes:
            if o in self.inverse_weights:
                for i, w in self.inverse_weights[o].items():
                    g_target[i] = g_target.get(i, 0.0) + w

        g_current: Dict[int, float] = {}
        for o in current_out_spikes:
            if o in self.inverse_weights:
                for i, w in self.inverse_weights[o].items():
                    g_current[i] = g_current.get(i, 0.0) + w

        in_set = set(current_in_spikes)
        combined: Dict[int, float] = {}
        all_ids = set(g_target.keys()) | set(g_current.keys()) | in_set
        
        for i in all_ids:
            val = g_target.get(i, 0.0) - g_current.get(i, 0.0)
            if i in in_set:
                val += 1.0
            combined[i] = val

        return [i for i, v in combined.items() if v > threshold]