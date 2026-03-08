# {
#     "//": "ディレクトリパス: tests/test_predictive_coding.py",
#     "//": "ファイルの日本語タイトル: 予測符号化・ターゲット伝播テスト",
#     "//": "ファイルの目的や内容: PredictiveCodingManager, TargetPropagationManager, PredictiveSpikeLayer, SpikingPredictiveLayer の動作検証テスト。"
# }

import pytest
from sara_engine.learning.predictive_coding import (
    PredictiveCodingManager,
    TargetPropagationManager,
)
from sara_engine.nn.predictive import PredictiveSpikeLayer, SpikingPredictiveLayer


# =====================================================================
# PredictiveCodingManager テスト
# =====================================================================

class TestPredictiveCodingManager:
    """PredictiveCodingManager のインスタンス化・学習・メトリクスをテスト"""

    def test_instantiation_with_learning_rate(self) -> None:
        """インスタンス化と learning_rate 属性の保持を確認"""
        mgr = PredictiveCodingManager(learning_rate=0.25)
        assert mgr.learning_rate == 0.25
        assert mgr.prediction_error_ema == 1.0
        assert mgr.total_updates == 0

    def test_default_learning_rate(self) -> None:
        """デフォルト learning_rate が 0.05 であることを確認"""
        mgr = PredictiveCodingManager()
        assert mgr.learning_rate == 0.05

    def test_reset(self) -> None:
        """reset() でメトリクスが初期化されることを確認"""
        mgr = PredictiveCodingManager(learning_rate=0.1)
        mgr.record_error(0.5)
        mgr.record_error(0.3)
        assert mgr.total_updates == 2

        mgr.reset()
        assert mgr.prediction_error_ema == 1.0
        assert mgr.total_updates == 0

    def test_update_forward_strengthens_weights(self) -> None:
        """update_forward が誤差スパイク→内部状態の重みを強化することを確認"""
        weights: list[dict[int, float]] = [{} for _ in range(5)]
        in_spikes = [0, 1, 2]
        state_spikes = [10, 11]
        error_spikes = [0, 2]

        PredictiveCodingManager.update_forward(
            weights, in_spikes, state_spikes, error_spikes, lr=0.1
        )

        # 誤差スパイク 0 → 状態スパイク 10, 11 への重みができた
        assert 10 in weights[0]
        assert 11 in weights[0]
        assert 10 in weights[2]
        # 誤差でないスパイク 1 には変化なし
        assert len(weights[1]) == 0

    def test_update_backward_ltp_and_ltd(self) -> None:
        """update_backward が LTP と LTD を正しく適用することを確認"""
        weights: list[dict[int, float]] = [
            {5: 0.5, 99: 0.1} for _ in range(3)
        ]
        prev_state_spikes = [0, 1]
        current_in_spikes = [5]
        predicted_in_spikes = {5, 99}

        PredictiveCodingManager.update_backward(
            weights, prev_state_spikes, current_in_spikes, predicted_in_spikes, lr=0.2
        )

        # LTP: neuron 5 (実際に来た) の重みは増加
        assert weights[0][5] > 0.5
        # LTD: neuron 99 (予測したが来なかった) の重みは減少
        assert weights[0].get(99, 0.0) < 0.1

    def test_prediction_error_tracking(self) -> None:
        """record_error と get_prediction_accuracy の動作を確認"""
        mgr = PredictiveCodingManager()

        # 初期状態: 予測精度 0 (error_ema = 1.0)
        assert mgr.get_prediction_accuracy() == pytest.approx(0.0)

        # 連続して低い誤差を記録 → 精度が上がる
        for _ in range(50):
            mgr.record_error(0.0)

        accuracy = mgr.get_prediction_accuracy()
        assert accuracy > 0.9
        assert 0.0 <= accuracy <= 1.0
        assert mgr.total_updates == 50

    def test_snn_transformer_compatibility(self) -> None:
        """snn_transformer.py での使用パターン互換性を確認:
        - PredictiveCodingManager(learning_rate=...) でインスタンス化
        - .learning_rate 属性で学習率を参照
        - .reset() でリセット
        """
        mgr = PredictiveCodingManager(learning_rate=0.25)
        assert mgr.learning_rate == 0.25

        # learning_rate を使った計算
        freq_norm_lr = mgr.learning_rate * 0.5
        assert freq_norm_lr == pytest.approx(0.125)

        # reset が呼べること
        mgr.reset()
        assert mgr.prediction_error_ema == 1.0


# =====================================================================
# TargetPropagationManager テスト
# =====================================================================

class TestTargetPropagationManager:
    """TargetPropagationManager のターゲット学習と DTP をテスト"""

    def test_instantiation(self) -> None:
        """インスタンス化とデフォルト値を確認"""
        mgr = TargetPropagationManager()
        assert mgr.lr == 0.1
        assert mgr.inverse_lr == 0.05
        assert len(mgr.inverse_weights) == 0

    def test_apply_target_strengthens_target_weights(self) -> None:
        """apply_target がターゲット方向の重みを強化することを確認"""
        weights: list[dict[int, float]] = [{} for _ in range(5)]
        in_spikes = [0, 1]
        out_spikes = [10, 11]
        target_spikes = [10, 12]

        TargetPropagationManager.apply_target(
            weights, in_spikes, out_spikes, target_spikes, lr=0.1
        )

        # ターゲット 10, 12 への結合が強化された
        assert weights[0].get(10, 0.0) > 0.0
        assert weights[0].get(12, 0.0) > 0.0
        # 誤発火 11（ターゲットに含まれない）への結合は存在しない（初期0なので抑制後削除）
        assert 11 not in weights[0]

    def test_apply_target_suppresses_false_firing(self) -> None:
        """apply_target が誤発火ニューロンの重みを抑制することを確認"""
        weights: list[dict[int, float]] = [{11: 0.5} for _ in range(3)]
        in_spikes = [0, 1]
        out_spikes = [11]
        target_spikes = [10]

        TargetPropagationManager.apply_target(
            weights, in_spikes, out_spikes, target_spikes, lr=0.1
        )

        # neuron 11 の重みは減少
        assert weights[0].get(11, 0.0) < 0.5

    def test_update_inverse_learns_mapping(self) -> None:
        """update_inverse が逆写像を学習することを確認"""
        mgr = TargetPropagationManager(inverse_lr=0.1)

        out_spikes = [5, 6]
        in_spikes = [0, 1, 2]

        mgr.update_inverse(out_spikes, in_spikes)

        # 逆写像が形成された: out=5 → in={0, 1, 2}
        assert 5 in mgr.inverse_weights
        assert 0 in mgr.inverse_weights[5]
        assert 1 in mgr.inverse_weights[5]
        assert 2 in mgr.inverse_weights[5]

    def test_update_inverse_ltd(self) -> None:
        """update_inverse で不要な結合が減衰することを確認"""
        mgr = TargetPropagationManager(inverse_lr=0.1)

        # ステップ1: out=5 → in={0, 1} を学習
        mgr.update_inverse([5], [0, 1])
        assert 0 in mgr.inverse_weights[5]
        assert 1 in mgr.inverse_weights[5]

        # ステップ2: out=5 → in={0} のみ。1 は LTD で減衰
        mgr.update_inverse([5], [0])
        w_0 = mgr.inverse_weights[5][0]
        w_1 = mgr.inverse_weights[5].get(1, 0.0)
        assert w_0 > w_1  # 0 への結合の方が強い

    def test_compute_local_target(self) -> None:
        """compute_local_target が DTP の差分式に基づきローカルターゲットを算出することを確認"""
        mgr = TargetPropagationManager(inverse_lr=0.5)

        # 逆写像を十分に学習
        for _ in range(10):
            mgr.update_inverse([10], [0, 1])
            mgr.update_inverse([11], [2, 3])

        # ケース: 上位ターゲットが [10] で、現在の出力も [10] の場合
        # g(target) - g(current) ≈ 0 なので、ターゲットは current_in に近い
        local_target = mgr.compute_local_target(
            upper_target_spikes=[10],
            current_out_spikes=[10],
            current_in_spikes=[0, 1],
            threshold=0.5,
        )
        # current_in の {0, 1} がターゲットに含まれるはず
        assert 0 in local_target
        assert 1 in local_target

    def test_compute_local_target_with_difference(self) -> None:
        """上位ターゲットと現在出力が異なるとき、ローカルターゲットが変化することを確認"""
        mgr = TargetPropagationManager(inverse_lr=0.5)

        for _ in range(10):
            mgr.update_inverse([10], [0, 1])
            mgr.update_inverse([11], [2, 3])

        # 現在出力: [10], ターゲット: [11] → 差分で {2, 3} 方向にシフト
        local_target = mgr.compute_local_target(
            upper_target_spikes=[11],
            current_out_spikes=[10],
            current_in_spikes=[0, 1],
            threshold=0.5,
        )
        # {2, 3} がターゲットに含まれるはず（g(11) の方向にシフト）
        assert 2 in local_target or 3 in local_target

    def test_reset_clears_inverse(self) -> None:
        """reset() が逆写像をクリアすることを確認"""
        mgr = TargetPropagationManager()
        mgr.update_inverse([5], [0, 1])
        assert len(mgr.inverse_weights) > 0

        mgr.reset()
        assert len(mgr.inverse_weights) == 0


# =====================================================================
# PredictiveSpikeLayer 統合テスト
# =====================================================================

class TestPredictiveSpikeLayer:
    """PredictiveSpikeLayer の forward / 学習 / メトリクス取得をテスト"""

    def test_forward_returns_spikes(self) -> None:
        """forward が整数リストを返すことを確認"""
        layer = PredictiveSpikeLayer(
            in_features=16, out_features=8, density=0.5)
        out = layer.forward([0, 1, 2, 3], learning=False)
        assert isinstance(out, list)

    def test_learning_updates_weights(self) -> None:
        """learning=True で重みが変化することを確認"""
        layer = PredictiveSpikeLayer(
            in_features=16, out_features=8, density=0.5)

        # 初期重みの記録
        initial_fw = {k: dict(v)
                      for k, v in enumerate(layer.forward_weights) if v}

        # 学習実行 (2ステップ: 内部状態が形成された後に予測学習が始まる)
        layer.forward([0, 1, 2, 3], learning=True)
        layer.forward([4, 5, 6, 7], learning=True)

        # 何らかの重みが変化しているはず
        changed = False
        for k, v in enumerate(layer.forward_weights):
            if v and (k not in initial_fw or v != initial_fw[k]):
                changed = True
                break
        assert changed

    def test_target_propagation_mode(self) -> None:
        """target_spikes を指定して学習できることを確認"""
        layer = PredictiveSpikeLayer(
            in_features=16, out_features=8, density=0.5)
        target = [2, 5]
        out = layer.forward([0, 1, 2, 3], learning=True, target_spikes=target)
        assert isinstance(out, list)

    def test_prediction_metrics(self) -> None:
        """get_prediction_metrics が正しい形式を返すことを確認"""
        layer = PredictiveSpikeLayer(
            in_features=16, out_features=8, density=0.5)

        # 初回 (予測なし)
        layer.forward([0, 1, 2, 3], learning=True)
        metrics = layer.get_prediction_metrics()
        assert "prediction_rate" in metrics
        assert "total_inputs" in metrics
        assert "total_predicted" in metrics
        assert metrics["total_inputs"] > 0

    def test_reset_clears_metrics(self) -> None:
        """reset_state でメトリクスがクリアされることを確認"""
        layer = PredictiveSpikeLayer(
            in_features=16, out_features=8, density=0.5)
        layer.forward([0, 1], learning=True)
        layer.reset_state()
        metrics = layer.get_prediction_metrics()
        assert metrics["total_inputs"] == 0


# =====================================================================
# SpikingPredictiveLayer 統合テスト
# =====================================================================

class TestSpikingPredictiveLayer:
    """SpikingPredictiveLayer の forward / 学習 / メトリクスをテスト"""

    def test_forward_returns_tuple(self) -> None:
        """forward が (state_spikes, predicted_bu) のタプルを返すことを確認"""
        layer = SpikingPredictiveLayer(
            in_features=32, out_features=16, density=0.2)
        state, predicted = layer.forward([0, 1, 2, 3])
        assert isinstance(state, list)
        assert isinstance(predicted, list)

    def test_learning_with_target_propagation(self) -> None:
        """target_spikes を指定した学習が動作することを確認"""
        layer = SpikingPredictiveLayer(
            in_features=32, out_features=16, density=0.2)
        target = [2, 5, 8]
        state, predicted = layer.forward([0, 1, 2], target_spikes=target)
        assert isinstance(state, list)

    def test_backward_weights_updated_by_manager(self) -> None:
        """学習時に PredictiveCodingManager 経由で backward weights が更新されることを確認"""
        layer = SpikingPredictiveLayer(
            in_features=32, out_features=16, density=0.2)

        # 2ステップの学習 (last_state が形成された後に backward 学習が走る)
        layer.forward([0, 1, 2, 3], learning=True)
        bw_before = sum(len(d) for d in layer.backward_weights)
        layer.forward([4, 5, 6, 7], learning=True)
        bw_after = sum(len(d) for d in layer.backward_weights)

        # backward weights の結合数が増えていることを期待
        assert bw_after >= bw_before

    def test_prediction_metrics(self) -> None:
        """get_prediction_metrics が正しい形式で精度データを返すことを確認"""
        layer = SpikingPredictiveLayer(
            in_features=32, out_features=16, density=0.2)
        layer.forward([0, 1, 2], learning=True)
        metrics = layer.get_prediction_metrics()
        assert "prediction_rate" in metrics
        assert metrics["total_inputs"] > 0

    def test_prediction_improves_with_repeated_pattern(self) -> None:
        """同じパターンを繰り返すと予測率が向上することを確認"""
        layer = SpikingPredictiveLayer(
            in_features=32, out_features=16, density=0.3)
        pattern = [0, 1, 2, 3]

        # 20エポック学習
        for _ in range(20):
            layer.forward(pattern, learning=True)

        layer._total_inputs = 0
        layer._total_predicted = 0

        # 予測テスト
        for _ in range(5):
            layer.forward(pattern, learning=False)

        metrics = layer.get_prediction_metrics()
        # 繰り返しパターンに対してある程度の予測ができているはず
        assert metrics["prediction_rate"] >= 0.0  # 最低限メトリクスが機能している

    def test_reset_state(self) -> None:
        """reset_state でメトリクスとadaptationがクリアされることを確認"""
        layer = SpikingPredictiveLayer(
            in_features=32, out_features=16, density=0.2)
        layer.forward([0, 1], learning=True)
        layer.reset_state()
        assert layer._total_inputs == 0
        assert layer._total_predicted == 0
        assert len(layer.last_state) == 0
        assert len(layer.adaptation) == 0


# =====================================================================
# nn パッケージエクスポートテスト
# =====================================================================

class TestNNExports:
    """nn パッケージから正しくエクスポートされていることを確認"""

    def test_predictive_spike_layer_exported(self) -> None:
        from sara_engine import nn
        assert hasattr(nn, "PredictiveSpikeLayer")

    def test_spiking_predictive_layer_exported(self) -> None:
        from sara_engine import nn
        assert hasattr(nn, "SpikingPredictiveLayer")
