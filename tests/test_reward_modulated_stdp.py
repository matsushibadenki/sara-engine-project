# ディレクトリパス: tests/test_reward_modulated_stdp.py
# ファイルの日本語タイトル: 報酬修飾型STDP テストスイート
# ファイルの目的や内容:
#   DopamineSignalModel, EligibilityTraceManager, RewardModulatedSTDPManager,
#   ThreeFactorLearningManager, RewardModulatedLinearSpike の包括的ユニットテスト。
#   強化学習タスクでの収束性テスト (統合テスト) を含む。

import random

from sara_engine.learning.reward_modulated_stdp import (
    DopamineSignalModel,
    EligibilityTraceManager,
    RewardModulatedSTDPManager,
)
from sara_engine.learning.three_factor_learning import ThreeFactorLearningManager
from sara_engine.nn.rstdp import RewardModulatedLinearSpike


# ====================================================================
# DopamineSignalModel のテスト
# ====================================================================

class TestDopamineSignalModel:
    """ドーパミン信号モデルのユニットテスト。"""

    def test_init_default(self) -> None:
        """デフォルト初期化で正しい初期値が設定される。"""
        model = DopamineSignalModel()
        assert model.tonic_level == 0.0
        assert model.phasic_level == 0.0
        assert model.reward_baseline == 0.0
        assert model.get_signal() == 0.0

    def test_deliver_positive_reward(self) -> None:
        """正の報酬でフェーズィック成分が正になる。"""
        model = DopamineSignalModel()
        signal = model.deliver_reward(1.0)
        assert signal > 0.0
        assert model.last_rpe > 0.0

    def test_deliver_negative_reward(self) -> None:
        """負の報酬でフェーズィック成分が負になる。"""
        model = DopamineSignalModel()
        signal = model.deliver_reward(-1.0)
        assert signal < 0.0
        assert model.last_rpe < 0.0

    def test_rpe_decreases_with_consistent_rewards(self) -> None:
        """一貫した報酬を受け続けると、RPEが減少する（予測が適応する）。"""
        model = DopamineSignalModel(baseline_decay=0.9)
        rpe_values = []
        for _ in range(20):
            model.deliver_reward(1.0)
            rpe_values.append(abs(model.last_rpe))
            model.phasic_level = 0.0  # 次のRPE計算のためリセット

        # 最初のRPEは最後のRPEより大きいはず
        assert rpe_values[0] > rpe_values[-1]

    def test_phasic_decay(self) -> None:
        """フェーズィック成分が時間経過で自然減衰する。"""
        model = DopamineSignalModel(tau_dopamine=10.0)
        model.deliver_reward(1.0)
        initial_phasic = model.phasic_level

        for _ in range(50):
            model.step()

        assert abs(model.phasic_level) < abs(initial_phasic)

    def test_surprise_signal(self) -> None:
        """ベースラインが適応した後の予想外の報酬でRPEが大きくなる。"""
        model = DopamineSignalModel(baseline_decay=0.9)
        # ベースラインを正の報酬に適応させる
        for _ in range(50):
            model.deliver_reward(1.0)
            model.phasic_level = 0.0

        # 突然の負の報酬 → 大きな負のRPE
        model.deliver_reward(-1.0)
        assert model.last_rpe < -0.5

    def test_reset(self) -> None:
        """リセットで全内部状態がクリアされる。"""
        model = DopamineSignalModel()
        model.deliver_reward(1.0)
        model.reset()
        assert model.phasic_level == 0.0
        assert model.reward_baseline == 0.0
        assert model.get_signal() == 0.0

    def test_state_dict_roundtrip(self) -> None:
        """state_dict / load_state_dict の往復一貫性。"""
        model = DopamineSignalModel()
        model.deliver_reward(1.0)
        state = model.state_dict()

        model2 = DopamineSignalModel()
        model2.load_state_dict(state)
        assert model2.phasic_level == model.phasic_level
        assert model2.reward_baseline == model.reward_baseline


# ====================================================================
# EligibilityTraceManager のテスト
# ====================================================================

class TestEligibilityTraceManager:
    """適格度トレース管理のユニットテスト。"""

    def test_init(self) -> None:
        """初期状態でトレースが空。"""
        mgr = EligibilityTraceManager()
        assert len(mgr.ltp_traces) == 0
        assert len(mgr.ltd_traces) == 0

    def test_causal_pair_creates_ltp(self) -> None:
        """因果的ペア (pre と post が同時発火) で LTP トレースが生成される。"""
        mgr = EligibilityTraceManager(a_plus=1.0)
        mgr.update_traces(pre_spikes=[0], post_spikes=[5], current_time=1.0)
        assert (0, 5) in mgr.ltp_traces
        assert mgr.ltp_traces[(0, 5)] > 0.0

    def test_net_trace_positive_for_causal(self) -> None:
        """因果的発火で正味のトレースが正。"""
        mgr = EligibilityTraceManager()
        mgr.update_traces(pre_spikes=[0], post_spikes=[5], current_time=1.0)
        net = mgr.get_net_trace(0, 5)
        assert net > 0.0

    def test_trace_decay(self) -> None:
        """トレースが時間経過で減衰する。"""
        mgr = EligibilityTraceManager(tau_eligibility=5.0, a_plus=10.0)
        mgr.update_traces(pre_spikes=[0], post_spikes=[5], current_time=1.0)
        initial = mgr.ltp_traces[(0, 5)]

        for _ in range(20):
            mgr.decay_traces()

        if (0, 5) in mgr.ltp_traces:
            assert mgr.ltp_traces[(0, 5)] < initial
        else:
            # 1e-6以下で削除されたケース
            pass

    def test_multiple_spikes_accumulate(self) -> None:
        """複数回の発火でトレースが蓄積する。"""
        mgr = EligibilityTraceManager(a_plus=1.0, tau_eligibility=100.0)
        mgr.update_traces(pre_spikes=[0], post_spikes=[5], current_time=1.0)
        first = mgr.ltp_traces[(0, 5)]
        mgr.update_traces(pre_spikes=[0], post_spikes=[5], current_time=2.0)
        second = mgr.ltp_traces[(0, 5)]
        assert second > first

    def test_get_all_traces(self) -> None:
        """全トレースの取得が正しい。"""
        mgr = EligibilityTraceManager()
        mgr.update_traces(pre_spikes=[0, 1], post_spikes=[
                          5, 6], current_time=1.0)
        traces = mgr.get_all_traces()
        assert len(traces) > 0
        for key, val in traces.items():
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_reset(self) -> None:
        """リセットで全データがクリアされる。"""
        mgr = EligibilityTraceManager()
        mgr.update_traces(pre_spikes=[0], post_spikes=[5], current_time=1.0)
        mgr.reset()
        assert len(mgr.ltp_traces) == 0
        assert len(mgr.ltd_traces) == 0
        assert len(mgr.last_spike_time) == 0

    def test_state_dict_roundtrip(self) -> None:
        """state_dict / load_state_dict の往復一貫性。"""
        mgr = EligibilityTraceManager()
        mgr.update_traces(pre_spikes=[0, 1], post_spikes=[5], current_time=1.0)
        state = mgr.state_dict()

        mgr2 = EligibilityTraceManager()
        mgr2.load_state_dict(state)
        assert mgr2.ltp_traces == mgr.ltp_traces


# ====================================================================
# RewardModulatedSTDPManager のテスト
# ====================================================================

class TestRewardModulatedSTDPManager:
    """R-STDP統合マネージャーのユニットテスト。"""

    def test_init(self) -> None:
        """初期化が正常に完了する。"""
        mgr = RewardModulatedSTDPManager()
        stats = mgr.get_stats()
        assert stats["total_updates"] == 0.0
        assert stats["cumulative_reward"] == 0.0

    def test_record_and_reward_positive(self) -> None:
        """正の報酬で因果的トレースを持つシナプスが強化される。"""
        mgr = RewardModulatedSTDPManager(learning_rate=0.1, w_max=5.0)
        weights: list[dict[int, float]] = [{1: 1.0}, {}, {}]

        mgr.record_spikes(pre_spikes=[0], post_spikes=[1], current_time=1.0)
        mgr.deliver_reward(1.0)
        count = mgr.apply_weight_updates(weights)

        assert count > 0
        # 正の報酬 + 因果的トレース → 重み増加
        assert weights[0][1] > 1.0

    def test_record_and_reward_negative(self) -> None:
        """負の報酬で因果的トレースを持つシナプスが弱化される。"""
        mgr = RewardModulatedSTDPManager(learning_rate=0.1, w_max=5.0)
        weights: list[dict[int, float]] = [{1: 1.0}, {}, {}]

        mgr.record_spikes(pre_spikes=[0], post_spikes=[1], current_time=1.0)
        mgr.deliver_reward(-1.0)
        count = mgr.apply_weight_updates(weights)

        assert count > 0
        # 負の報酬 + 因果的トレース → 重み減少
        assert weights[0].get(1, 0.0) < 1.0

    def test_soft_bound_constraint(self) -> None:
        """Soft-bound制約で重みが上限を超えない。"""
        mgr = RewardModulatedSTDPManager(
            learning_rate=1.0, w_max=3.0, a_plus=5.0
        )
        weights: list[dict[int, float]] = [{1: 2.9}, {}, {}]

        mgr.record_spikes(pre_spikes=[0], post_spikes=[1], current_time=1.0)
        mgr.deliver_reward(10.0)
        mgr.apply_weight_updates(weights)

        assert weights[0][1] <= 3.0

    def test_no_update_without_trace(self) -> None:
        """トレースがない場合は更新されない。"""
        mgr = RewardModulatedSTDPManager()
        weights: list[dict[int, float]] = [{1: 1.0}]

        mgr.deliver_reward(1.0)
        count = mgr.apply_weight_updates(weights)
        assert count == 0

    def test_step_decays_traces_and_dopamine(self) -> None:
        """step() でトレースとドーパミンが減衰する。"""
        mgr = RewardModulatedSTDPManager(tau_eligibility=5.0, tau_dopamine=5.0)
        mgr.record_spikes(pre_spikes=[0], post_spikes=[1], current_time=1.0)
        mgr.deliver_reward(1.0)

        initial_signal = mgr.dopamine.get_signal()
        for _ in range(50):
            mgr.step()

        assert abs(mgr.dopamine.get_signal()) < abs(initial_signal)

    def test_homeostasis(self) -> None:
        """ホメオスタシスが重み合計を目標値に近づける。"""
        mgr = RewardModulatedSTDPManager(
            homeostatic_target=5.0, homeostatic_rate=0.5
        )
        weights: list[dict[int, float]] = [{0: 5.0, 1: 5.0}]

        mgr.record_spikes(pre_spikes=[0], post_spikes=[0, 1], current_time=1.0)
        mgr.deliver_reward(0.0)  # 報酬なし、ホメオスタシスのみ作用
        mgr.apply_weight_updates(weights)

        total = sum(weights[0].values())
        # ホメオスタシスにより目標値 (5.0) に近づく方向へ
        assert total < 10.0

    def test_state_dict_roundtrip(self) -> None:
        """state_dict / load_state_dict の往復一貫性。"""
        mgr = RewardModulatedSTDPManager()
        mgr.record_spikes(pre_spikes=[0], post_spikes=[1], current_time=1.0)
        mgr.deliver_reward(1.0)
        state = mgr.state_dict()

        mgr2 = RewardModulatedSTDPManager()
        mgr2.load_state_dict(state)
        assert mgr2.cumulative_reward == mgr.cumulative_reward

    def test_reset(self) -> None:
        """リセットで全状態がクリアされる。"""
        mgr = RewardModulatedSTDPManager()
        mgr.record_spikes(pre_spikes=[0], post_spikes=[1], current_time=1.0)
        mgr.deliver_reward(1.0)
        mgr.reset()
        stats = mgr.get_stats()
        assert stats["total_updates"] == 0.0
        assert stats["cumulative_reward"] == 0.0


# ====================================================================
# ThreeFactorLearningManager のテスト
# ====================================================================

class TestThreeFactorLearningManager:
    """三要素学習マネージャーのユニットテスト。"""

    def test_init(self) -> None:
        """デフォルト初期化。"""
        mgr = ThreeFactorLearningManager()
        assert mgr.reward_baseline == 0.0
        assert mgr.reward_count == 0

    def test_update_trace_positive(self) -> None:
        """正の strength で LTP トレースが蓄積される。"""
        mgr = ThreeFactorLearningManager()
        mgr.update_trace(0, 1, strength=1.0, time=0.0)
        assert mgr._ltp_traces[(0, 1)] > 0.0

    def test_update_trace_negative(self) -> None:
        """負の strength で LTD トレースが蓄積される。"""
        mgr = ThreeFactorLearningManager()
        mgr.update_trace(0, 1, strength=-1.0, time=0.0)
        assert mgr._ltd_traces[(0, 1)] > 0.0

    def test_apply_reward_basic(self) -> None:
        """基本的な報酬適用で重み更新量が返る。"""
        mgr = ThreeFactorLearningManager(use_rpe=False)
        mgr.update_trace(0, 1, strength=1.0, time=0.0)
        updates = mgr.apply_reward(1.0)
        assert (0, 1) in updates
        assert updates[(0, 1)] > 0.0

    def test_rpe_reduces_with_consistent_rewards(self) -> None:
        """RPEが一貫した報酬で減少する（ベースラインが報酬に適応する）。"""
        mgr = ThreeFactorLearningManager(
            baseline_decay=0.9, use_rpe=True, trace_decay=0.0
        )
        # 報酬を繰り返し与えてベースラインの適応を確認
        for _ in range(50):
            mgr.update_trace(0, 1, strength=1.0, time=0.0)
            mgr.apply_reward(1.0)

        # ベースラインが1.0に近づいているはず
        assert mgr.reward_baseline > 0.5
        # RPE (= 1.0 - baseline) は初期の1.0より小さいはず
        assert abs(1.0 - mgr.reward_baseline) < 0.5

    def test_decay_all_traces(self) -> None:
        """全トレースの減衰が正しく動く。"""
        mgr = ThreeFactorLearningManager(trace_decay=0.5)
        mgr.update_trace(0, 1, strength=1.0, time=0.0)
        initial = mgr._ltp_traces[(0, 1)]
        mgr.decay_all_traces()
        assert mgr._ltp_traces.get((0, 1), 0.0) < initial

    def test_state_dict_roundtrip(self) -> None:
        """state_dict / load_state_dict の往復。"""
        mgr = ThreeFactorLearningManager()
        mgr.update_trace(0, 1, strength=1.0, time=0.0)
        mgr.apply_reward(1.0)
        state = mgr.state_dict()

        mgr2 = ThreeFactorLearningManager()
        mgr2.load_state_dict(state)
        assert mgr2.reward_baseline == mgr.reward_baseline
        assert mgr2.reward_count == mgr.reward_count

    def test_reset(self) -> None:
        """リセットで全状態がクリアされる。"""
        mgr = ThreeFactorLearningManager()
        mgr.update_trace(0, 1, strength=1.0, time=0.0)
        mgr.apply_reward(1.0)
        mgr.reset()
        assert len(mgr._traces) == 0
        assert mgr.reward_baseline == 0.0


# ====================================================================
# RewardModulatedLinearSpike のテスト
# ====================================================================

class TestRewardModulatedLinearSpike:
    """R-STDP層のユニットテスト。"""

    def test_init(self) -> None:
        """初期化が正常に完了する。"""
        layer = RewardModulatedLinearSpike(in_features=4, out_features=3)
        assert layer.in_features == 4
        assert layer.out_features == 3

    def test_forward_produces_spikes(self) -> None:
        """forward で出力スパイクが生成される。"""
        layer = RewardModulatedLinearSpike(
            in_features=4, out_features=3, density=1.0
        )
        out = layer([0, 1, 2, 3], learning=False)
        assert isinstance(out, list)
        # 全入力が発火しているのでいくつかの出力が発火するはず
        assert len(out) > 0

    def test_learning_creates_traces(self) -> None:
        """learning=True でトレースが記録される。"""
        layer = RewardModulatedLinearSpike(
            in_features=4, out_features=3, density=1.0, epsilon=0.0
        )
        layer([0, 1], learning=True)
        # LTPトレースが記録されているはず
        assert len(layer.ltp_traces) > 0 or len(layer.ltd_traces) > 0

    def test_apply_reward_modifies_weights(self) -> None:
        """報酬適用で重みが変化する。"""
        random.seed(42)
        layer = RewardModulatedLinearSpike(
            in_features=4, out_features=3, density=1.0, epsilon=0.0
        )
        # 初期重みを記録
        initial_weights = {
            (i, t): w
            for i in range(4)
            for t, w in layer.weights[i].items()
        }

        # 学習ステップ
        layer([0, 1], learning=True)
        layer.apply_reward(1.0, learning_rate=0.5)

        # 重みが変化しているか確認
        changed = False
        for i in range(4):
            for t, w in layer.weights[i].items():
                if initial_weights.get((i, t), w) != w:
                    changed = True
                    break
        assert changed

    def test_epsilon_greedy_exploration(self) -> None:
        """ε-greedy探索で追加のスパイクが生成される。"""
        random.seed(42)
        layer = RewardModulatedLinearSpike(
            in_features=4, out_features=10, density=0.1, epsilon=1.0
        )
        # epsilon=1.0 なので常に探索的スパイクが追加される
        out = layer([0], learning=True)
        # 探索によりスパイクが追加されているはず
        assert len(out) >= 1

    def test_epsilon_decay(self) -> None:
        """apply_reward後にεが減衰する。"""
        layer = RewardModulatedLinearSpike(
            in_features=4, out_features=3, epsilon=0.5, epsilon_decay=0.9
        )
        initial_eps = layer.epsilon
        layer([0], learning=True)
        layer.apply_reward(1.0)
        assert layer.epsilon < initial_eps

    def test_weights_stay_bounded(self) -> None:
        """重みがw_maxを超えない。"""
        layer = RewardModulatedLinearSpike(
            in_features=2, out_features=2, density=1.0, w_max=3.0
        )
        for _ in range(100):
            layer([0, 1], learning=True)
            layer.apply_reward(1.0, learning_rate=0.5)

        for i in range(2):
            for w in layer.weights[i].values():
                assert w <= 3.0

    def test_eligibility_traces_property(self) -> None:
        """後方互換性のeligibility_tracesプロパティが動作する。"""
        layer = RewardModulatedLinearSpike(
            in_features=4, out_features=3, density=1.0, epsilon=0.0
        )
        layer([0, 1], learning=True)
        traces = layer.eligibility_traces
        assert isinstance(traces, dict)

    def test_eligibility_traces_setter(self) -> None:
        """eligibility_tracesのセッターが動作する。"""
        layer = RewardModulatedLinearSpike(in_features=4, out_features=3)
        layer.eligibility_traces = {(0, 1): 1.0, (1, 2): -0.5}
        assert (0, 1) in layer.ltp_traces
        assert (1, 2) in layer.ltd_traces

    def test_get_stats(self) -> None:
        """統計情報が取得できる。"""
        layer = RewardModulatedLinearSpike(in_features=4, out_features=3)
        stats = layer.get_stats()
        assert "epsilon" in stats
        assert "reward_baseline" in stats

    def test_reset_state(self) -> None:
        """reset_stateでトレースがクリアされる。"""
        layer = RewardModulatedLinearSpike(
            in_features=4, out_features=3, density=1.0
        )
        layer([0, 1], learning=True)
        layer.reset_state()
        assert len(layer.ltp_traces) == 0
        assert len(layer.ltd_traces) == 0


# ====================================================================
# 統合テスト: 強化学習タスク
# ====================================================================

class TestRLIntegration:
    """強化学習タスクでのR-STDPの収束性テスト。"""

    def test_simple_rl_convergence(self) -> None:
        """簡単なRL問題（状態→行動マッピング）でR-STDPが収束する。

        State 0 → Action 1 が正解, State 1 → Action 2 が正解。
        """
        random.seed(42)
        layer = RewardModulatedLinearSpike(
            in_features=2,
            out_features=3,
            density=1.0,
            epsilon=0.3,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            w_max=5.0,
        )

        def get_reward(state: int, action: int) -> float:
            if state == 0 and action == 1:
                return 1.0
            elif state == 1 and action == 2:
                return 1.0
            else:
                return -1.0

        correct_count = 0
        total_count = 0

        for epoch in range(300):
            state = random.choice([0, 1])
            out_spikes = layer([state], learning=True)

            if not out_spikes:
                action = random.choice([0, 1, 2])
                # 手動でトレースを追加
                layer.ltp_traces[(state, action)] = (
                    layer.ltp_traces.get((state, action), 0.0) + 1.0
                )
            else:
                action = out_spikes[0]

            reward = get_reward(state, action)
            layer.apply_reward(reward, learning_rate=0.2)

            # 最後の100エピソードの正答率を計算
            if epoch >= 200:
                total_count += 1
                expected = 1 if state == 0 else 2
                if action == expected:
                    correct_count += 1

        # 最低40%の正答率を期待（ランダムは33%）
        accuracy = correct_count / max(1, total_count)
        assert accuracy > 0.4, f"Accuracy too low: {accuracy:.2%}"

    def test_rstdp_manager_rl_task(self) -> None:
        """RewardModulatedSTDPManagerを使った強化学習タスク。"""
        random.seed(123)
        mgr = RewardModulatedSTDPManager(
            learning_rate=0.1,
            w_max=5.0,
            baseline_decay=0.95,
            a_plus=1.0,
            a_minus=0.3,
        )
        # 2入力、3出力のシナプス
        weights: list[dict[int, float]] = [
            {0: 0.5, 1: 0.5, 2: 0.5},
            {0: 0.5, 1: 0.5, 2: 0.5},
        ]

        for epoch in range(200):
            state = random.choice([0, 1])
            # 状態を入力スパイクとして与える
            pre_spikes = [state]
            # 簡易的な膜電位計算
            potentials = [0.0, 0.0, 0.0]
            for s in pre_spikes:
                for t, w in weights[s].items():
                    potentials[t] += w
            # 最大ポテンシャルの出力を選択
            post_spikes = [potentials.index(max(potentials))]

            mgr.record_spikes(pre_spikes, post_spikes, float(epoch))

            # 報酬判定
            expected = 1 if state == 0 else 2
            reward = 1.0 if post_spikes[0] == expected else -1.0

            mgr.deliver_reward(reward)
            mgr.apply_weight_updates(weights)
            mgr.step()

        stats = mgr.get_stats()
        assert stats["total_updates"] > 0
        assert stats["cumulative_reward"] != 0.0
