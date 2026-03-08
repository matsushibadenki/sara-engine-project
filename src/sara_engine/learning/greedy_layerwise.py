# ディレクトリパス: src/sara_engine/learning/greedy_layerwise.py
# ファイルの日本語タイトル: 貪欲な層ごとの教師なし学習マネージャー
# ファイルの目的や内容: 深層SNNの各層を下位層から順に独立して教師なし学習し、
#   学習済みの層を凍結して次の層を学習する貪欲な積み上げ戦略を管理する。

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =====================================================================
# プロトコル: Greedy Layer-wise 学習可能な層
# =====================================================================


@runtime_checkable
class GreedyTrainableLayer(Protocol):
    """Greedy Layer-wise Learning で学習可能な層のプロトコル。"""

    def forward(self, in_spikes: List[int], learning: bool = False) -> List[int]:
        """順伝播。"""
        ...

    def freeze(self) -> None:
        """学習を凍結する。"""
        ...

    def unfreeze(self) -> None:
        """凍結を解除する。"""
        ...

    @property
    def is_frozen(self) -> bool:
        """凍結状態を返す。"""
        ...


# =====================================================================
# 学習メトリクス
# =====================================================================


@dataclass
class LayerTrainingMetrics:
    """各層の学習メトリクスを保持するデータクラス。"""

    layer_index: int = 0
    total_steps: int = 0
    converged: bool = False
    convergence_step: int = -1
    final_stability: float = 0.0
    firing_rate_history: List[float] = field(default_factory=list)
    stability_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式でメトリクスを返す。"""
        return {
            "layer_index": self.layer_index,
            "total_steps": self.total_steps,
            "converged": self.converged,
            "convergence_step": self.convergence_step,
            "final_stability": self.final_stability,
            "firing_rate_history": list(self.firing_rate_history),
            "stability_history": list(self.stability_history),
        }


# =====================================================================
# メインクラス: GreedyLayerWiseTrainer
# =====================================================================


class GreedyLayerWiseTrainer:
    """貪欲な層ごとの教師なし学習トレーナー。

    Hinton et al. (2006) の Deep Belief Networks で提唱された
    Greedy Layer-wise Unsupervised Pre-training を SNN に適応。

    学習戦略:
        1. 最下位層から順に教師なし学習を実行
        2. 収束判定後、学習済みの層を凍結
        3. 凍結済み層を通してデータを伝播し、次層の入力とする
        4. 全層の学習が完了するまで繰り返す

    収束判定:
        出力スパイクパターンの Jaccard 類似度が閾値を超えた状態が
        一定ステップ続いたら収束と判定する。
    """

    def __init__(
        self,
        epochs_per_layer: int = 5,
        max_steps_per_epoch: int = 100,
        convergence_threshold: float = 0.85,
        convergence_patience: int = 10,
        log_interval: int = 20,
    ) -> None:
        """
        Args:
            epochs_per_layer: 各層あたりの学習エポック数。
            max_steps_per_epoch: 各エポックの最大ステップ数。
            convergence_threshold: 収束判定のJaccard類似度閾値。
            convergence_patience: 収束状態を維持する必要がある連続ステップ数。
            log_interval: ログ出力間隔（ステップ数）。
        """
        self.epochs_per_layer = epochs_per_layer
        self.max_steps_per_epoch = max_steps_per_epoch
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.log_interval = log_interval

        # 各層のメトリクス
        self._layer_metrics: List[LayerTrainingMetrics] = []

    # ------------------------------------------------------------------
    # パブリック API
    # ------------------------------------------------------------------

    def train_layer(
        self,
        layer: GreedyTrainableLayer,
        data_iterator: Iterable[List[int]],
        layer_index: int = 0,
    ) -> LayerTrainingMetrics:
        """単一層に対して教師なし学習を実行する。

        Args:
            layer: 学習対象の層（GreedyTrainableLayer プロトコルに適合）。
            data_iterator: 入力スパイクパターンのイテラブル。
            layer_index: メトリクス記録用の層インデックス。

        Returns:
            学習メトリクス。
        """
        metrics = LayerTrainingMetrics(layer_index=layer_index)
        prev_spikes: Optional[List[int]] = None
        patience_counter = 0
        epoch_source = self._ensure_reiterable(data_iterator)

        layer.unfreeze()

        for epoch in range(self.epochs_per_layer):
            if metrics.converged:
                break

            for step, input_spikes in enumerate(epoch_source):
                if step >= self.max_steps_per_epoch:
                    break

                # 順伝播（学習有効）
                out_spikes = layer.forward(input_spikes, learning=True)
                metrics.total_steps += 1

                # --- 発火率の記録 ---
                if hasattr(layer, "out_features"):
                    n_out = getattr(layer, "out_features")
                    rate = len(out_spikes) / max(1, n_out)
                else:
                    rate = float(len(out_spikes))
                metrics.firing_rate_history.append(rate)

                # --- 収束判定 (Jaccard 類似度) ---
                stability = self._compute_stability(prev_spikes, out_spikes)
                metrics.stability_history.append(stability)
                prev_spikes = list(out_spikes)

                if stability >= self.convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= self.convergence_patience:
                        metrics.converged = True
                        metrics.convergence_step = metrics.total_steps
                        metrics.final_stability = stability
                        logger.info(
                            "Layer %d converged at step %d (stability=%.3f)",
                            layer_index,
                            metrics.total_steps,
                            stability,
                        )
                        break
                else:
                    patience_counter = 0

                # --- ログ出力 ---
                if metrics.total_steps % self.log_interval == 0:
                    logger.debug(
                        "Layer %d | step %d | stability=%.3f | fire_rate=%.3f",
                        layer_index,
                        metrics.total_steps,
                        stability,
                        rate,
                    )

        if not metrics.converged:
            metrics.final_stability = (
                metrics.stability_history[-1] if metrics.stability_history else 0.0
            )

        # 学習完了 → 凍結
        layer.freeze()
        return metrics

    def train_stack(
        self,
        layers: List[GreedyTrainableLayer],
        data_iterator_factory: Any,
    ) -> List[LayerTrainingMetrics]:
        """複数層を下位層から順にGreedy学習する。

        Args:
            layers: 学習対象の層リスト（下位から上位の順）。
            data_iterator_factory: 呼び出すたびに新しいデータイテレータを返す
                                   callable。各エポックで新しいイテレータを取得する。

        Returns:
            全層のメトリクスリスト。
        """
        all_metrics: List[LayerTrainingMetrics] = []
        frozen_layers: List[GreedyTrainableLayer] = []

        for idx, layer in enumerate(layers):
            logger.info("=== Training Layer %d / %d ===", idx + 1, len(layers))

            # 凍結済み層を通してデータを伝播するイテレータを作成
            propagated_iterator = self._create_propagated_iterator(
                frozen_layers, data_iterator_factory
            )

            # 単一層を学習
            metrics = self.train_layer(
                layer, propagated_iterator, layer_index=idx)
            all_metrics.append(metrics)

            # 学習済み層を凍結リストに追加
            frozen_layers.append(layer)

            logger.info(
                "Layer %d: converged=%s, steps=%d, stability=%.3f",
                idx,
                metrics.converged,
                metrics.total_steps,
                metrics.final_stability,
            )

        self._layer_metrics = all_metrics
        return all_metrics

    def get_metrics(self) -> List[LayerTrainingMetrics]:
        """全層の学習メトリクスを返す。"""
        return list(self._layer_metrics)

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """トレーナーの状態辞書を返す。"""
        return {
            "epochs_per_layer": self.epochs_per_layer,
            "max_steps_per_epoch": self.max_steps_per_epoch,
            "convergence_threshold": self.convergence_threshold,
            "convergence_patience": self.convergence_patience,
            "log_interval": self.log_interval,
            "layer_metrics": [m.to_dict() for m in self._layer_metrics],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """トレーナーの状態を復元する。"""
        self.epochs_per_layer = int(
            state.get("epochs_per_layer", self.epochs_per_layer))
        self.max_steps_per_epoch = int(
            state.get("max_steps_per_epoch", self.max_steps_per_epoch)
        )
        self.convergence_threshold = float(
            state.get("convergence_threshold", self.convergence_threshold)
        )
        self.convergence_patience = int(
            state.get("convergence_patience", self.convergence_patience)
        )
        self.log_interval = int(state.get("log_interval", self.log_interval))

        self._layer_metrics = []
        for m_dict in state.get("layer_metrics", []):
            metrics = LayerTrainingMetrics(
                layer_index=int(m_dict.get("layer_index", 0)),
                total_steps=int(m_dict.get("total_steps", 0)),
                converged=bool(m_dict.get("converged", False)),
                convergence_step=int(m_dict.get("convergence_step", -1)),
                final_stability=float(m_dict.get("final_stability", 0.0)),
                firing_rate_history=list(
                    m_dict.get("firing_rate_history", [])),
                stability_history=list(m_dict.get("stability_history", [])),
            )
            self._layer_metrics.append(metrics)

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stability(
        prev_spikes: Optional[List[int]],
        current_spikes: List[int],
    ) -> float:
        """2つのスパイクパターン間の Jaccard 類似度を計算する。

        Args:
            prev_spikes: 前ステップのスパイクパターン。None の場合は 0.0 を返す。
            current_spikes: 現ステップのスパイクパターン。

        Returns:
            Jaccard 類似度 [0.0, 1.0]。
        """
        if prev_spikes is None:
            return 0.0

        set_prev = set(prev_spikes)
        set_curr = set(current_spikes)
        union = set_prev | set_curr

        if not union:
            return 1.0  # 両方空なら一致

        intersection = set_prev & set_curr
        return len(intersection) / len(union)

    @staticmethod
    def _create_propagated_iterator(
        frozen_layers: List[GreedyTrainableLayer],
        data_iterator_factory: Any,
    ) -> Iterator[List[int]]:
        """凍結済み層を通してデータを伝播するジェネレータ。

        Args:
            frozen_layers: 凍結済み層のリスト。
            data_iterator_factory: 生データイテレータのファクトリ。

        Yields:
            凍結層を通過した後のスパイクパターン。
        """
        raw_iterator = data_iterator_factory()
        for raw_spikes in raw_iterator:
            spikes = raw_spikes
            for layer in frozen_layers:
                spikes = layer.forward(spikes, learning=False)
            yield spikes

    @staticmethod
    def _ensure_reiterable(
        data_iterator: Iterable[List[int]],
    ) -> Iterable[List[int]]:
        """エポックごとに再走査できる入力へ正規化する。

        `iter(obj) is obj` となる単発イテレータが渡された場合は、
        1回だけリストへ退避して以後のエポックでも再利用できるようにする。
        """
        iterator = iter(data_iterator)
        if iterator is data_iterator:
            return [list(spikes) for spikes in iterator]
        return data_iterator
