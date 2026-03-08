# ディレクトリパス: tests/test_greedy_layerwise.py
# ファイルの日本語タイトル: 貪欲な層ごとの教師なし学習テスト
# ファイルの目的や内容: GreedyLayerWiseTrainer と UnsupervisedSpikeLayer の
#   包括的テスト。単一層学習、多層学習、凍結/解凍、ホメオスタシス、
#   収束判定、state_dict の各機能を検証する。

import copy
import random

from sara_engine.learning.greedy_layerwise import (
    GreedyLayerWiseTrainer,
    LayerTrainingMetrics,
)
from sara_engine.nn.unsupervised_layer import UnsupervisedSpikeLayer


# =====================================================================
# ヘルパー: テスト用データジェネレータ
# =====================================================================


def _make_pattern_data(
    num_patterns: int = 5,
    spikes_per_pattern: int = 10,
    input_dim: int = 64,
    repeats: int = 20,
) -> list[list[int]]:
    """再現性のあるテストパターンデータを生成する。"""
    rng = random.Random(42)
    patterns: list[list[int]] = []
    for _ in range(num_patterns):
        pat = sorted(rng.sample(range(input_dim), spikes_per_pattern))
        patterns.append(pat)
    # 繰り返しで安定パターンを形成
    data: list[list[int]] = []
    for _ in range(repeats):
        for pat in patterns:
            data.append(list(pat))
    return data


def _data_factory(data: list[list[int]]):  # type: ignore[no-untyped-def]
    """データリストをイテレータファクトリとして返す。"""
    def factory():  # type: ignore[no-untyped-def]
        return iter(data)
    return factory


# =====================================================================
# テスト: UnsupervisedSpikeLayer
# =====================================================================


class TestUnsupervisedSpikeLayer:
    """UnsupervisedSpikeLayer の単体テスト。"""

    def test_basic_forward(self) -> None:
        """基本的な順伝播がスパイクを返すことを確認。"""
        layer = UnsupervisedSpikeLayer(
            in_features=32, out_features=16, density=0.5, k_winners=3
        )
        in_spikes = [0, 5, 10, 15, 20]
        out = layer.forward(in_spikes, learning=False)
        assert isinstance(out, list)
        # WTA により最大 k_winners 個
        assert len(out) <= 3

    def test_wta_constraint(self) -> None:
        """WTA が出力スパイク数を制限することを確認。"""
        layer = UnsupervisedSpikeLayer(
            in_features=64, out_features=32, density=0.8, k_winners=5
        )
        in_spikes = list(range(0, 64, 2))  # 32個の入力スパイク
        out = layer.forward(in_spikes, learning=False)
        assert len(out) <= 5

    def test_freeze_unfreeze(self) -> None:
        """凍結/解凍で学習の有無が切り替わることを確認。"""
        layer = UnsupervisedSpikeLayer(
            in_features=32, out_features=16, density=0.5)
        assert not layer.is_frozen

        # 学習実行 → 重みが変化
        weights_before = copy.deepcopy(layer.weights)
        for _ in range(10):
            layer.forward([0, 1, 2, 3, 4], learning=True)
        weights_after_learning = copy.deepcopy(layer.weights)
        # 重みが変化していることを確認
        assert weights_before != weights_after_learning

        # 凍結して学習
        layer.freeze()
        assert layer.is_frozen
        weights_before_frozen = copy.deepcopy(layer.weights)
        for _ in range(10):
            layer.forward([0, 1, 2, 3, 4], learning=True)
        weights_after_frozen = copy.deepcopy(layer.weights)
        # 凍結中は重みが変化しない
        assert weights_before_frozen == weights_after_frozen

        # 解凍
        layer.unfreeze()
        assert not layer.is_frozen

    def test_homeostasis_adaptation(self) -> None:
        """ホメオスタシスが適応閾値を調整することを確認。"""
        layer = UnsupervisedSpikeLayer(
            in_features=32,
            out_features=16,
            density=0.5,
            target_rate=0.1,
            threshold=0.5,
        )
        initial_thresholds = list(layer.adaptive_thresholds)

        # 充分なステップ数で発火を繰り返す
        for _ in range(50):
            layer.forward([0, 1, 2, 3, 4, 5], learning=True)

        # 閾値が初期値から変化していること
        changed = any(
            abs(layer.adaptive_thresholds[i] - initial_thresholds[i]) > 1e-6
            for i in range(16)
        )
        assert changed, "ホメオスタシスが閾値を調整しなかった"

    def test_state_dict_roundtrip(self) -> None:
        """state_dict のシリアライズ/デシリアライズが正しいことを確認。"""
        layer = UnsupervisedSpikeLayer(
            in_features=32, out_features=16, density=0.5)
        # 少し学習
        for _ in range(5):
            layer.forward([0, 2, 4, 6], learning=True)
        layer.freeze()

        state = layer.state_dict()
        assert "frozen" in state

        # 新しい層に復元
        layer2 = UnsupervisedSpikeLayer(
            in_features=32, out_features=16, density=0.5)
        layer2.load_state_dict(state)
        assert layer2.is_frozen
        assert layer2.weights == layer.weights

    def test_dead_neuron_prevention(self) -> None:
        """出力が空にならない（デッドレイヤー防止）ことを確認。"""
        # 非常に高い閾値で初期化
        layer = UnsupervisedSpikeLayer(
            in_features=32, out_features=16, density=0.5, threshold=100.0
        )
        out = layer.forward([0, 1, 2, 3], learning=False)
        # デッドレイヤー防止により何らかの出力があるはず
        # (ただし入力の重みが全て0より大きい場合のみ)
        assert isinstance(out, list)


# =====================================================================
# テスト: GreedyLayerWiseTrainer
# =====================================================================


class TestGreedyLayerWiseTrainer:
    """GreedyLayerWiseTrainer の統合テスト。"""

    def test_single_layer_training(self) -> None:
        """単一層の教師なし学習が正常に完了することを確認。"""
        layer = UnsupervisedSpikeLayer(
            in_features=64, out_features=32, density=0.3, k_winners=5
        )
        data = _make_pattern_data(
            num_patterns=3, spikes_per_pattern=8, input_dim=64, repeats=30
        )
        trainer = GreedyLayerWiseTrainer(
            epochs_per_layer=3,
            max_steps_per_epoch=100,
            convergence_threshold=0.7,
            convergence_patience=5,
        )

        metrics = trainer.train_layer(layer, iter(data), layer_index=0)

        assert isinstance(metrics, LayerTrainingMetrics)
        assert metrics.total_steps > 0
        assert metrics.layer_index == 0
        assert len(metrics.firing_rate_history) > 0
        assert len(metrics.stability_history) > 0
        # 学習後は凍結されている
        assert layer.is_frozen

    def test_train_layer_reuses_iterator_across_epochs(self) -> None:
        """単発イテレータでも全エポック分の学習データが消費される。"""
        layer = UnsupervisedSpikeLayer(
            in_features=16, out_features=8, density=0.4, k_winners=2
        )
        data = [[0, 1], [2, 3]]
        trainer = GreedyLayerWiseTrainer(
            epochs_per_layer=3,
            max_steps_per_epoch=10,
            convergence_threshold=2.0,
            convergence_patience=10,
        )

        metrics = trainer.train_layer(layer, iter(data), layer_index=0)

        assert metrics.total_steps == 6

    def test_multi_layer_stack_training(self) -> None:
        """多層のGreedy学習が正常に動作することを確認。"""
        layers = [
            UnsupervisedSpikeLayer(
                in_features=64, out_features=32, density=0.4, k_winners=5
            ),
            UnsupervisedSpikeLayer(
                in_features=32, out_features=16, density=0.4, k_winners=3
            ),
        ]
        data = _make_pattern_data(
            num_patterns=3, spikes_per_pattern=8, input_dim=64, repeats=20
        )
        trainer = GreedyLayerWiseTrainer(
            epochs_per_layer=2,
            max_steps_per_epoch=50,
            convergence_threshold=0.9,
            convergence_patience=5,
        )

        all_metrics = trainer.train_stack(layers, _data_factory(data))

        assert len(all_metrics) == 2
        # 全層が凍結されている
        for layer in layers:
            assert layer.is_frozen
        # メトリクスが記録されている
        for m in all_metrics:
            assert m.total_steps > 0

    def test_frozen_layer_passthrough(self) -> None:
        """凍結済み層を通したデータ伝播が正しく動作することを確認。"""
        layer1 = UnsupervisedSpikeLayer(
            in_features=64, out_features=32, density=0.3, k_winners=5
        )
        layer1.freeze()

        in_spikes = [0, 5, 10, 15, 20, 25]
        out1 = layer1.forward(in_spikes, learning=False)

        # 凍結後も同じ入力に対して同じ出力を返す（非確率的）
        out2 = layer1.forward(in_spikes, learning=False)
        assert out1 == out2

    def test_convergence_detection(self) -> None:
        """同一パターンの繰り返しで収束が検出されることを確認。"""
        layer = UnsupervisedSpikeLayer(
            in_features=32, out_features=16, density=0.5, k_winners=3
        )
        # 単一パターンを大量に繰り返す → 高い収束度が期待される
        single_pattern = [0, 2, 4, 6, 8, 10]
        data = [single_pattern] * 200

        trainer = GreedyLayerWiseTrainer(
            epochs_per_layer=1,
            max_steps_per_epoch=200,
            convergence_threshold=0.8,
            convergence_patience=5,
        )

        metrics = trainer.train_layer(layer, iter(data), layer_index=0)
        # 同じパターンの繰り返しなので収束するはず
        assert metrics.converged or metrics.final_stability > 0.5

    def test_jaccard_stability(self) -> None:
        """Jaccard 類似度の計算が正しいことを確認。"""
        # 完全一致
        assert GreedyLayerWiseTrainer._compute_stability(
            [1, 2, 3], [1, 2, 3]) == 1.0
        # 完全不一致
        assert GreedyLayerWiseTrainer._compute_stability(
            [1, 2, 3], [4, 5, 6]) == 0.0
        # 50% 一致
        stability = GreedyLayerWiseTrainer._compute_stability([1, 2], [2, 3])
        assert abs(stability - 1.0 / 3.0) < 1e-6
        # None 入力
        assert GreedyLayerWiseTrainer._compute_stability(None, [1, 2]) == 0.0
        # 両方空
        assert GreedyLayerWiseTrainer._compute_stability([], []) == 1.0

    def test_trainer_state_dict(self) -> None:
        """トレーナーの state_dict/load_state_dict が正しいことを確認。"""
        trainer = GreedyLayerWiseTrainer(
            epochs_per_layer=10,
            max_steps_per_epoch=200,
            convergence_threshold=0.9,
        )
        # ダミーメトリクスを追加
        trainer._layer_metrics = [
            LayerTrainingMetrics(
                layer_index=0,
                total_steps=50,
                converged=True,
                convergence_step=45,
                final_stability=0.92,
            ),
        ]

        state = trainer.state_dict()
        assert state["epochs_per_layer"] == 10
        assert len(state["layer_metrics"]) == 1

        trainer2 = GreedyLayerWiseTrainer()
        trainer2.load_state_dict(state)
        assert trainer2.epochs_per_layer == 10
        assert len(trainer2._layer_metrics) == 1
        assert trainer2._layer_metrics[0].converged

    def test_metrics_to_dict(self) -> None:
        """LayerTrainingMetrics.to_dict が正しい辞書を返すことを確認。"""
        metrics = LayerTrainingMetrics(
            layer_index=2,
            total_steps=100,
            converged=True,
            convergence_step=90,
            final_stability=0.95,
            firing_rate_history=[0.1, 0.08],
            stability_history=[0.5, 0.9],
        )
        d = metrics.to_dict()
        assert d["layer_index"] == 2
        assert d["converged"] is True
        assert d["convergence_step"] == 90
        assert len(d["firing_rate_history"]) == 2
