# {
#     "//": "ディレクトリパス: src/sara_engine/learning/predictive_coding.py",
#     "//": "ファイルの日本語タイトル: 予測符号化・ターゲット伝播モジュール",
#     "//": "ファイルの目的や内容: 誤差逆伝播法に依存せず、予測誤差に基づく局所学習(Predictive Coding)と目標発火に基づくTarget Propagationを提供する。インスタンス化対応、予測誤差追跡メトリクス、Difference Target Propagation (DTP) を実装。"
# }

from typing import List, Dict, Set, Optional

# --- 定数 ---
_WEIGHT_CAP: float = 3.0
_ERROR_EMA_ALPHA: float = 0.1


class PredictiveCodingManager:
    """予測符号化(Predictive Coding)に基づく局所的シナプス可塑性マネージャー

    インスタンスとして生成し、learning_rate や内部状態（予測誤差追跡）を保持する。
    static メソッドとしても呼び出し可能で、後方互換性を維持する。
    """

    def __init__(self, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate
        # --- 予測誤差追跡メトリクス ---
        self.prediction_error_ema: float = 1.0  # 初期値は「完全に予測不能」
        self.total_updates: int = 0

    def reset(self) -> None:
        """内部の追跡メトリクスをリセットする。"""
        self.prediction_error_ema = 1.0
        self.total_updates = 0

    def get_prediction_accuracy(self) -> float:
        """現在の予測精度を 0.0（全く予測不能）〜 1.0（完全予測）で返す。"""
        return max(0.0, min(1.0, 1.0 - self.prediction_error_ema))

    def record_error(self, error_rate: float) -> None:
        """予測誤差率を EMA で追跡する。

        Args:
            error_rate: 0.0（予測完全一致）〜 1.0（全て予測ミス）の誤差率。
        """
        clamped = max(0.0, min(1.0, error_rate))
        self.prediction_error_ema = (
            (1.0 - _ERROR_EMA_ALPHA) * self.prediction_error_ema
            + _ERROR_EMA_ALPHA * clamped
        )
        self.total_updates += 1

    # -----------------------------------------------------------------
    # ボトムアップ結合の学習
    # -----------------------------------------------------------------
    @staticmethod
    def update_forward(
        forward_weights: List[Dict[int, float]],
        in_spikes: List[int],
        state_spikes: List[int],
        error_spikes: List[int],
        lr: float = 0.05,
    ) -> None:
        """予測できなかった誤差(Surprise)から現在の内部状態への結合を強化する。"""
        state_set = set(state_spikes)
        for s in error_spikes:
            if s < len(forward_weights):
                for t in state_set:
                    forward_weights[s][t] = min(
                        _WEIGHT_CAP, forward_weights[s].get(t, 0.0) + lr
                    )

    # -----------------------------------------------------------------
    # トップダウン結合の学習
    # -----------------------------------------------------------------
    @staticmethod
    def update_backward(
        backward_weights: List[Dict[int, float]],
        prev_state_spikes: List[int],
        current_in_spikes: List[int],
        predicted_in_spikes: Set[int],
        lr: float = 0.1,
    ) -> None:
        """過去の状態から現在の入力への予測を強化し、誤予測を減衰させる。"""
        in_set = set(current_in_spikes)

        # 1. 実際の入力に対する予測結合の強化 (LTP)
        for s in prev_state_spikes:
            if s < len(backward_weights):
                for t in in_set:
                    backward_weights[s][t] = min(
                        _WEIGHT_CAP, backward_weights[s].get(t, 0.0) + lr
                    )

        # 2. 予測したのに来なかった入力への結合の減衰 (LTD)
        false_positives = predicted_in_spikes - in_set
        for s in prev_state_spikes:
            if s < len(backward_weights):
                for t in false_positives:
                    if t in backward_weights[s]:
                        backward_weights[s][t] -= lr * 0.5
                        if backward_weights[s][t] <= 0:
                            del backward_weights[s][t]


class TargetPropagationManager:
    """ターゲット伝播(Target Propagation)マネージャー

    標準的な Target Propagation に加え、Difference Target Propagation (DTP) を実装。
    DTP では逆写像(インバース関数)を学習し、より正確なローカルターゲットを算出する。
    """

    def __init__(self, lr: float = 0.1, inverse_lr: float = 0.05) -> None:
        self.lr = lr
        self.inverse_lr = inverse_lr
        # 逆写像の重み: {out_neuron: {in_neuron: weight}}
        self.inverse_weights: Dict[int, Dict[int, float]] = {}

    def reset(self) -> None:
        """逆写像の重みをクリアする。"""
        self.inverse_weights.clear()

    # -----------------------------------------------------------------
    # 標準 Target Propagation
    # -----------------------------------------------------------------
    @staticmethod
    def apply_target(
        weights: List[Dict[int, float]],
        in_spikes: List[int],
        out_spikes: List[int],
        target_spikes: List[int],
        lr: float = 0.1,
    ) -> None:
        """上位層から指定された目標発火パターンに近づくように重みを直接更新する。"""
        target_set = set(target_spikes)
        out_set = set(out_spikes)

        for s in in_spikes:
            if s < len(weights):
                # ターゲットとして指定されたニューロンへの結合を強化
                for t in target_set:
                    weights[s][t] = min(
                        _WEIGHT_CAP, weights[s].get(t, 0.0) + lr
                    )

                # 誤って発火したニューロンへの結合を抑制 (罰則)
                for t in out_set - target_set:
                    if t in weights[s]:
                        weights[s][t] -= lr
                        if weights[s][t] <= 0:
                            del weights[s][t]

    # -----------------------------------------------------------------
    # Difference Target Propagation (DTP)
    # -----------------------------------------------------------------
    def update_inverse(
        self,
        out_spikes: List[int],
        in_spikes: List[int],
        lr: Optional[float] = None,
    ) -> None:
        """逆写像 g(y) ≈ x を学習する。

        出力ニューロン → 入力ニューロン方向のマッピングを、
        ヘビアン的に LTP/LTD で更新する。

        Args:
            out_spikes: 順方向の出力スパイク（逆写像の入力に相当）。
            in_spikes:  順方向の入力スパイク（逆写像が再構成すべき目標）。
            lr: 学習率。省略時は self.inverse_lr を使用。
        """
        actual_lr = lr if lr is not None else self.inverse_lr
        in_set = set(in_spikes)

        for o in out_spikes:
            if o not in self.inverse_weights:
                self.inverse_weights[o] = {}
            w_map = self.inverse_weights[o]

            # LTP: 実際の入力ニューロンへの結合を強化
            for i in in_set:
                w_map[i] = min(_WEIGHT_CAP, w_map.get(i, 0.0) + actual_lr)

            # LTD: 結合はあるが実際に発火しなかった入力への減衰
            for i in list(w_map.keys()):
                if i not in in_set:
                    w_map[i] -= actual_lr * 0.3
                    if w_map[i] <= 0:
                        del w_map[i]

    def compute_local_target(
        self,
        upper_target_spikes: List[int],
        current_out_spikes: List[int],
        current_in_spikes: List[int],
        threshold: float = 0.5,
    ) -> List[int]:
        """Difference Target Propagation により現在層のローカルターゲットを算出する。

        DTP の核心: target_local = g(target_upper) - g(current_out) + current_in
        これにより、逆写像の系統的バイアスが相殺され、より正確な目標が得られる。

        Args:
            upper_target_spikes: 上位層から伝播されたターゲット発火パターン。
            current_out_spikes:  現在の出力スパイク。
            current_in_spikes:   現在の入力スパイク。
            threshold: ニューロンをターゲットとみなす電位閾値。

        Returns:
            ローカルターゲット発火のスパイクIDリスト。
        """
        # g(target_upper): 上位ターゲットを逆写像で入力空間に復元
        g_target: Dict[int, float] = {}
        for o in upper_target_spikes:
            if o in self.inverse_weights:
                for i, w in self.inverse_weights[o].items():
                    g_target[i] = g_target.get(i, 0.0) + w

        # g(current_out): 現在の出力を逆写像で入力空間に復元
        g_current: Dict[int, float] = {}
        for o in current_out_spikes:
            if o in self.inverse_weights:
                for i, w in self.inverse_weights[o].items():
                    g_current[i] = g_current.get(i, 0.0) + w

        # DTP: difference = g(target) - g(current) + current_in
        in_set = set(current_in_spikes)
        combined: Dict[int, float] = {}
        all_ids = set(g_target.keys()) | set(g_current.keys()) | in_set
        for i in all_ids:
            val = g_target.get(i, 0.0) - g_current.get(i, 0.0)
            if i in in_set:
                val += 1.0
            combined[i] = val

        # 閾値を超えたニューロンをターゲットとして返す
        return [i for i, v in combined.items() if v > threshold]
