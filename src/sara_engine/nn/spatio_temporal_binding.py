# ディレクトリパス: src/sara_engine/nn/spatio_temporal_binding.py
# ファイルの日本語タイトル: 時空間バインディング層
# ファイルの目的や内容: 複数のモダリティから非同期に入力されるスパイク（イベント）を、適格度トレース (Eligibility Traces) を用いて時間差を考慮しながら関連付ける機能を提供する。

from typing import List, Dict
from .module import SNNModule


class SpatioTemporalBindingLayer(SNNModule):
    """
    Spatio-Temporal Binding Layer for Multimodal Learning.
    Uses Eligibility Traces to associate asynchronous spikes across different modalities.
    """

    def __init__(
        self,
        modality_names: List[str],
        dim_per_modality: int,
        trace_decay: float = 0.9,
        learning_rate: float = 0.1,
        max_weight: float = 3.0,
        prune_threshold: float = 0.05,
    ):
        """
        Args:
            modality_names: 統合するモダリティ名のリスト (例: ['vision', 'language'])
            dim_per_modality: 各モダリティの入力次元数 (スパイク空間のサイズ)
            trace_decay: 適格度トレースの減衰率 (0.0=即座に忘却, 1.0=減衰なし)
            learning_rate: STDPライクな学習の学習率
            max_weight: 結合重みの上限値 (Soft-bound用)
            prune_threshold: 刈り込みを行う重みの閾値
        """
        super().__init__()
        self.modality_names = modality_names
        self.dim_per_modality = dim_per_modality
        self.trace_decay = trace_decay
        self.learning_rate = learning_rate
        self.max_weight = max_weight
        self.prune_threshold = prune_threshold

        # 各モダリティのニューロンごとの適格度トレース
        # {modality: {neuron_id: trace_value}}
        self.traces: Dict[str, Dict[int, float]] = {
            mod: {} for mod in modality_names}

        # モダリティ間の結合重み (方向付き)
        # {source_mod: {target_mod: {src_id: {tgt_id: weight}}}}
        self.weights: Dict[str, Dict[str, Dict[int, Dict[int, float]]]] = {}
        for src in modality_names:
            self.weights[src] = {}
            for tgt in modality_names:
                if src != tgt:
                    self.weights[src][tgt] = {}

        self.register_state("traces")
        self.register_state("weights")

    def forward(
        self, inputs: Dict[str, List[int]], learning: bool = False, threshold: float = 0.5
    ) -> Dict[str, List[int]]:
        """
        入力スパイクを処理し、1.トレースの更新、2.想起(Recall)、3.学習(LTP/LTD) を行う。

        Args:
            inputs: 各モダリティの入力スパイク {modality: [spike_indices]}
            learning: 学習を有効にするかどうか
            threshold: 想起のための発火閾値

        Returns:
            各モダリティで想起(Recall)されたスパイクの辞書
        """
        self._update_traces(inputs)

        # 1. 相互想起 (Recall) : あるモダリティの入力から別のモダリティを予測
        recalls: Dict[str, List[int]] = {mod: []
                                         for mod in self.modality_names}
        potentials: Dict[str, Dict[int, float]] = {
            mod: {} for mod in self.modality_names}

        # 全入力モダリティからの影響を集積
        for src_mod, spikes in inputs.items():
            if src_mod not in self.modality_names:
                continue
            for tgt_mod in self.modality_names:
                if src_mod == tgt_mod:
                    continue
                # src_mod -> tgt_mod の影響を計算
                for s in spikes:
                    if s in self.weights[src_mod][tgt_mod]:
                        for t, w in self.weights[src_mod][tgt_mod][s].items():
                            potentials[tgt_mod][t] = potentials[tgt_mod].get(
                                t, 0.0) + w

        # 閾値判定による発火
        for mod in self.modality_names:
            recalls[mod] = [k for k, v in potentials[mod].items() if v >
                            threshold]

        # 2. 学習 (Spatio-Temporal Learning)
        if learning:
            self._learn_spatio_temporal(inputs)
            self._prune_weights()

        return recalls

    def _update_traces(self, inputs: Dict[str, List[int]]) -> None:
        """全モダリティのトレースを指数減衰させ、新規入力を加算する"""
        for mod in self.modality_names:
            # 減衰
            for nid in list(self.traces[mod].keys()):
                self.traces[mod][nid] *= self.trace_decay
                if self.traces[mod][nid] < 0.01:
                    del self.traces[mod][nid]

            # 新規入力の加算 (最高値を1.0とする)
            if mod in inputs:
                for s in inputs[mod]:
                    self.traces[mod][s] = 1.0

    def _learn_spatio_temporal(self, inputs: Dict[str, List[int]]) -> None:
        """
        最近発火した(トレースが残っている)過去のスパイクと、現在の入力スパイクを結合する。
        """
        for src_mod in self.modality_names:
            for tgt_mod in self.modality_names:
                if src_mod == tgt_mod:
                    continue

                # src_mod の現在のスパイク と tgt_mod の過去のトレース
                # つまり tgt_mod -> src_mod の方向の結合を強化 (tgt_modが先、src_modが後)
                if src_mod in inputs:
                    for s in inputs[src_mod]:
                        # tgt_mod のトレースを走査
                        for t, trace_val in self.traces[tgt_mod].items():
                            if trace_val > 0.05:  # 有意なトレースがある場合のみ
                                # tgt_modのt -> src_modのs の結合
                                if t not in self.weights[tgt_mod][src_mod]:
                                    self.weights[tgt_mod][src_mod][t] = {}

                                current_w = self.weights[tgt_mod][src_mod][t].get(
                                    s, 0.0)
                                # Soft-bound な重み更新: 上限に近づくほど更新幅が小さくなる
                                delta = self.learning_rate * trace_val * \
                                    ((self.max_weight - current_w) / self.max_weight)
                                self.weights[tgt_mod][src_mod][t][s] = min(
                                    self.max_weight, current_w + delta)

    def _prune_weights(self) -> None:
        """使用されていない非常に小さな重みを削除する(忘却とリソース最適化)"""
        for src_mod in self.modality_names:
            for tgt_mod in self.modality_names:
                if src_mod == tgt_mod:
                    continue

                empty_sources: List[int] = []
                for s, targets in self.weights[src_mod][tgt_mod].items():
                    to_remove: List[int] = []
                    for t, w in targets.items():
                        # 常時わずかに減衰 (LTD / 忘却)
                        targets[t] *= 0.999
                        if targets[t] < self.prune_threshold:
                            to_remove.append(t)

                    for t in to_remove:
                        del targets[t]

                    if not targets:
                        empty_sources.append(s)

                for s in empty_sources:
                    del self.weights[src_mod][tgt_mod][s]

    def reset_state(self) -> None:
        """トレースをリセットする (重みは保持)"""
        super().reset_state()
        self.traces = {mod: {} for mod in self.modality_names}
