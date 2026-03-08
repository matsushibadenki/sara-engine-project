# ディレクトリパス: src/sara_engine/learning/reward_modulated_stdp.py
# ファイルの日本語タイトル: 報酬修飾型STDP (R-STDP) マネージャー
# ファイルの目的や内容:
#   生物学的に妥当な報酬修飾型STDP (Reward-Modulated STDP) を実装する。
#   ドーパミン信号モデル、適格度トレース管理、R-STDP統合マネージャーの3クラスで構成。
#   3要素学習則 Δw = η × e(t) × δ(t) による誤差逆伝播なしの強化学習を実現する。

import math
from typing import Dict, List, Tuple, Any, Optional


class DopamineSignalModel:
    """ドーパミン信号のダイナミクスモデル。

    トニック（基底）レベルとフェーズィック（報酬由来の速い応答）レベルを分離し、
    報酬予測誤差 (RPE: Reward Prediction Error) に基づくサプライズ信号を生成する。

    生物学的背景:
        - 中脳ドーパミンニューロン (VTA/SNc) は報酬の「予測誤差」を符号化する
        - 予想外の報酬 → フェーズィック発火増加 (δ > 0)
        - 予想通りの報酬 → 変化なし (δ ≈ 0)
        - 予想された報酬の欠如 → フェーズィック発火抑制 (δ < 0)
    """

    def __init__(
        self,
        tonic_level: float = 0.0,
        tau_dopamine: float = 20.0,
        baseline_decay: float = 0.99,
        rpe_scale: float = 1.0,
    ) -> None:
        """
        Args:
            tonic_level: ドーパミンのトニック(基底)レベル。
            tau_dopamine: フェーズィック信号の指数減衰時定数。
            baseline_decay: 報酬ベースライン(期待報酬)の EMA 減衰率。
            rpe_scale: RPE のスケーリング係数。
        """
        self.tonic_level = tonic_level
        self.tau_dopamine = tau_dopamine
        self.baseline_decay = baseline_decay
        self.rpe_scale = rpe_scale

        # フェーズィック成分（RPE由来の急速な応答）
        self.phasic_level: float = 0.0
        # 報酬ベースライン（指数移動平均で追跡する「期待報酬」）
        self.reward_baseline: float = 0.0
        # 直近のRPE値
        self.last_rpe: float = 0.0
        # 累計報酬回数（ベースライン安定判定用）
        self.reward_count: int = 0

    def deliver_reward(self, reward: float) -> float:
        """環境から報酬を受け取り、報酬予測誤差 (RPE) を計算してドーパミン信号を更新する。

        RPE: δ = r - V(s)  (報酬 - 期待報酬)
        正のδ → 予想以上の報酬 → シナプス強化を促進
        負のδ → 期待はずれ → シナプス弱化を促進

        Args:
            reward: 環境からの報酬スカラー値。

        Returns:
            このタイムステップのドーパミンシグナル（トニック + フェーズィック）。
        """
        # 報酬予測誤差の計算
        rpe = (reward - self.reward_baseline) * self.rpe_scale
        self.last_rpe = rpe

        # フェーズィック成分にRPEを加算
        self.phasic_level += rpe

        # 報酬ベースライン（期待報酬）の更新
        self.reward_count += 1
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline
            + (1.0 - self.baseline_decay) * reward
        )

        return self.get_signal()

    def step(self) -> float:
        """1タイムステップ進め、フェーズィック成分を自然減衰させる。

        Returns:
            現在のドーパミンシグナル。
        """
        # フェーズィック成分の指数減衰: d/dt = -phasic / τ
        if abs(self.phasic_level) > 1e-8:
            decay = math.exp(-1.0 /
                             self.tau_dopamine) if self.tau_dopamine > 0 else 0.0
            self.phasic_level *= decay
        else:
            self.phasic_level = 0.0

        return self.get_signal()

    def get_signal(self) -> float:
        """現在のドーパミンシグナル（トニック + フェーズィック）を返す。"""
        return self.tonic_level + self.phasic_level

    def reset(self) -> None:
        """内部状態をリセットする。"""
        self.phasic_level = 0.0
        self.reward_baseline = 0.0
        self.last_rpe = 0.0
        self.reward_count = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tonic_level": self.tonic_level,
            "phasic_level": self.phasic_level,
            "reward_baseline": self.reward_baseline,
            "last_rpe": self.last_rpe,
            "reward_count": self.reward_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.tonic_level = float(state.get("tonic_level", self.tonic_level))
        self.phasic_level = float(state.get("phasic_level", 0.0))
        self.reward_baseline = float(state.get("reward_baseline", 0.0))
        self.last_rpe = float(state.get("last_rpe", 0.0))
        self.reward_count = int(state.get("reward_count", 0))


class EligibilityTraceManager:
    """適格度トレース (Eligibility Trace) の管理。

    STDP窓関数に基づくシナプスペア (pre, post) のトレースを保持し、
    時間経過による指数減衰を適用する。

    生物学的背景:
        - シナプス可塑性は発火直後が最も大きく、時間とともに減衰する
        - この「痕跡」がドーパミン信号と掛け合わされることで、
          時間的に遅延した報酬でも適切なシナプスが強化/弱化される
        - LTPトレース: pre→postの因果的発火ペア (Δt > 0)
        - LTDトレース: post→preの逆因果的発火ペア (Δt < 0)
    """

    def __init__(
        self,
        tau_eligibility: float = 25.0,
        tau_stdp_plus: float = 20.0,
        tau_stdp_minus: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 0.5,
    ) -> None:
        """
        Args:
            tau_eligibility: 適格度トレースの指数減衰時定数。
            tau_stdp_plus: STDP窓関数の LTP 側の時定数。
            tau_stdp_minus: STDP窓関数の LTD 側の時定数。
            a_plus: LTP の最大振幅。
            a_minus: LTD の最大振幅。
        """
        self.tau_eligibility = tau_eligibility
        self.tau_stdp_plus = tau_stdp_plus
        self.tau_stdp_minus = tau_stdp_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

        # LTPトレース: {(pre_id, post_id): trace_value}
        self.ltp_traces: Dict[Tuple[int, int], float] = {}
        # LTDトレース: {(pre_id, post_id): trace_value}
        self.ltd_traces: Dict[Tuple[int, int], float] = {}
        # 各ニューロンの最終発火時刻
        self.last_spike_time: Dict[int, float] = {}

    def record_pre_spike(self, pre_id: int, time: float) -> None:
        """シナプス前ニューロンの発火を記録する。"""
        self.last_spike_time[pre_id] = time

    def record_post_spike(self, post_id: int, time: float) -> None:
        """シナプス後ニューロンの発火を記録する。"""
        self.last_spike_time[post_id] = time

    def update_traces(
        self,
        pre_spikes: List[int],
        post_spikes: List[int],
        current_time: float,
    ) -> None:
        """発火イベントに基づいてSTDP窓関数を適用し、トレースを更新する。

        STDP窓関数:
            Δt = t_post - t_pre
            Δt > 0 (因果的) → LTP:  Δe_+ = A+ × exp(-Δt / τ+)
            Δt < 0 (逆因果) → LTD:  Δe_- = A- × exp(Δt / τ-)

        Args:
            pre_spikes: 発火したシナプス前ニューロンのIDリスト。
            post_spikes: 発火したシナプス後ニューロンのIDリスト。
            current_time: 現在の時刻。
        """
        # シナプス前の発火を記録
        for pre_id in pre_spikes:
            self.record_pre_spike(pre_id, current_time)

        # シナプス後の発火を記録
        for post_id in post_spikes:
            self.record_post_spike(post_id, current_time)

        # 因果的ペア (pre fired, then post fired) → LTP トレースを蓄積
        for post_id in post_spikes:
            for pre_id in pre_spikes:
                # 同タイムステップ内で共に発火 → Δt ≈ 0, 最大の LTP
                key = (pre_id, post_id)
                current_trace = self.ltp_traces.get(key, 0.0)
                self.ltp_traces[key] = current_trace + self.a_plus

        # 逆因果的ペア: postが先に発火していたpreの未発火ペア → LTD
        # （簡略化: 前のタイムステップでpostが発火し、今preが発火した場合）
        for pre_id in pre_spikes:
            for post_id in self.last_spike_time:
                if post_id in post_spikes:
                    continue  # 今のステップで共に発火したペアはLTPで処理済み
                post_time = self.last_spike_time.get(post_id, -1000.0)
                delta_t = current_time - post_time
                # postが最近発火していた場合のみLTDを適用
                if 0.0 < delta_t < self.tau_stdp_minus * 3:
                    key = (pre_id, post_id)
                    ltd_amount = self.a_minus * \
                        math.exp(-delta_t / self.tau_stdp_minus)
                    current_trace = self.ltd_traces.get(key, 0.0)
                    self.ltd_traces[key] = current_trace + ltd_amount

    def decay_traces(self) -> None:
        """全トレースに時間減衰を適用する。"""
        decay_factor = math.exp(-1.0 /
                                self.tau_eligibility) if self.tau_eligibility > 0 else 0.0

        keys_to_remove: List[Tuple[int, int]] = []
        for key in self.ltp_traces:
            self.ltp_traces[key] *= decay_factor
            if self.ltp_traces[key] < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.ltp_traces[key]

        keys_to_remove = []
        for key in self.ltd_traces:
            self.ltd_traces[key] *= decay_factor
            if self.ltd_traces[key] < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.ltd_traces[key]

    def get_net_trace(self, pre_id: int, post_id: int) -> float:
        """特定のシナプスペアの正味のトレース値 (LTP - LTD) を返す。"""
        key = (pre_id, post_id)
        ltp = self.ltp_traces.get(key, 0.0)
        ltd = self.ltd_traces.get(key, 0.0)
        return ltp - ltd

    def get_all_traces(self) -> Dict[Tuple[int, int], float]:
        """全シナプスペアの正味のトレースを辞書で返す。"""
        all_keys = set(self.ltp_traces.keys()) | set(self.ltd_traces.keys())
        result: Dict[Tuple[int, int], float] = {}
        for key in all_keys:
            net = self.ltp_traces.get(key, 0.0) - self.ltd_traces.get(key, 0.0)
            if abs(net) > 1e-8:
                result[key] = net
        return result

    def reset(self) -> None:
        """全トレースをクリアする。"""
        self.ltp_traces.clear()
        self.ltd_traces.clear()
        self.last_spike_time.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ltp_traces": {f"{k[0]}_{k[1]}": v for k, v in self.ltp_traces.items()},
            "ltd_traces": {f"{k[0]}_{k[1]}": v for k, v in self.ltd_traces.items()},
            "last_spike_time": dict(self.last_spike_time),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ltp_traces.clear()
        for k, v in state.get("ltp_traces", {}).items():
            parts = str(k).split("_")
            if len(parts) == 2:
                self.ltp_traces[(int(parts[0]), int(parts[1]))] = float(v)
        self.ltd_traces.clear()
        for k, v in state.get("ltd_traces", {}).items():
            parts = str(k).split("_")
            if len(parts) == 2:
                self.ltd_traces[(int(parts[0]), int(parts[1]))] = float(v)
        self.last_spike_time = {
            int(k): float(v) for k, v in state.get("last_spike_time", {}).items()
        }


class RewardModulatedSTDPManager:
    """報酬修飾型STDP (R-STDP) 統合マネージャー。

    3要素学習則を統合し、シナプス重みの更新を管理する:

        Δw_ij = η × e_ij(t) × δ(t)

    ここで:
        η: 学習率
        e_ij(t): シナプス (i→j) の適格度トレース（STDP由来）
        δ(t): ドーパミン信号（報酬予測誤差）

    Soft-bound制約により重みの暴走を防止し、
    恒常性維持のためのシナプススケーリングも提供する。
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        w_min: float = 0.0,
        w_max: float = 3.0,
        tau_eligibility: float = 25.0,
        tau_dopamine: float = 20.0,
        tau_stdp_plus: float = 20.0,
        tau_stdp_minus: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 0.5,
        baseline_decay: float = 0.99,
        homeostatic_target: Optional[float] = None,
        homeostatic_rate: float = 0.001,
    ) -> None:
        """
        Args:
            learning_rate: 重み更新の学習率 η。
            w_min: シナプス重みの最小値。
            w_max: シナプス重みの最大値。
            tau_eligibility: 適格度トレースの時定数。
            tau_dopamine: ドーパミン信号の減衰時定数。
            tau_stdp_plus: STDP窓の LTP 時定数。
            tau_stdp_minus: STDP窓の LTD 時定数。
            a_plus: LTP の振幅。
            a_minus: LTD の振幅。
            baseline_decay: 報酬ベースラインの EMA 減衰率。
            homeostatic_target: 重み合計のホメオスタシス目標値。None で無効。
            homeostatic_rate: ホメオスタシス調整の強度。
        """
        self.learning_rate = learning_rate
        self.w_min = w_min
        self.w_max = w_max
        self.homeostatic_target = homeostatic_target
        self.homeostatic_rate = homeostatic_rate

        # サブコンポーネント
        self.dopamine = DopamineSignalModel(
            tau_dopamine=tau_dopamine,
            baseline_decay=baseline_decay,
        )
        self.eligibility = EligibilityTraceManager(
            tau_eligibility=tau_eligibility,
            tau_stdp_plus=tau_stdp_plus,
            tau_stdp_minus=tau_stdp_minus,
            a_plus=a_plus,
            a_minus=a_minus,
        )

        # 統計情報
        self.total_updates: int = 0
        self.cumulative_reward: float = 0.0

    def record_spikes(
        self,
        pre_spikes: List[int],
        post_spikes: List[int],
        current_time: float,
    ) -> None:
        """発火イベントを記録し STDP トレースを更新する。

        この時点では重みは変更しない。報酬到着を待つ。

        Args:
            pre_spikes: 発火したシナプス前ニューロンのIDリスト。
            post_spikes: 発火したシナプス後ニューロンのIDリスト。
            current_time: 現在の時刻。
        """
        self.eligibility.update_traces(pre_spikes, post_spikes, current_time)

    def deliver_reward(self, reward: float) -> float:
        """環境から報酬を受け取り、ドーパミン信号を更新する。

        Args:
            reward: 報酬スカラー値。

        Returns:
            ドーパミンシグナル（RPEに基づく）。
        """
        self.cumulative_reward += reward
        return self.dopamine.deliver_reward(reward)

    def step(self) -> None:
        """1タイムステップを進め、トレースとドーパミンを減衰させる。"""
        self.eligibility.decay_traces()
        self.dopamine.step()

    def compute_weight_updates(
        self,
        weights: List[Dict[int, float]],
    ) -> Dict[Tuple[int, int], float]:
        """3要素学習則 Δw = η × e(t) × δ(t) に基づく重み更新量を計算する。

        Soft-bound制約により、重みが上下限に近づくほど更新量が減衰する。

        Args:
            weights: weights[pre_id][post_id] = weight の形式のシナプス重み。

        Returns:
            {(pre_id, post_id): delta_w} の辞書。
        """
        dopamine_signal = self.dopamine.get_signal()
        all_traces = self.eligibility.get_all_traces()
        updates: Dict[Tuple[int, int], float] = {}

        for (pre_id, post_id), trace in all_traces.items():
            raw_delta = self.learning_rate * trace * dopamine_signal

            # Soft-bound制約
            if pre_id < len(weights) and post_id in weights[pre_id]:
                current_w = weights[pre_id][post_id]
            else:
                current_w = 0.0

            if raw_delta > 0:
                # LTP方向: 上限に近づくほど更新量を抑制
                soft_factor = (self.w_max - current_w) / self.w_max
            else:
                # LTD方向: 下限に近づくほど更新量を抑制
                soft_factor = (current_w - self.w_min) / self.w_max

            soft_factor = max(0.0, min(1.0, soft_factor))
            updates[(pre_id, post_id)] = raw_delta * soft_factor

        return updates

    def apply_weight_updates(
        self,
        weights: List[Dict[int, float]],
    ) -> int:
        """重み更新を計算し、実際にシナプスに適用する。

        Args:
            weights: weights[pre_id][post_id] = weight の形式のシナプス重み。

        Returns:
            更新されたシナプスの数。
        """
        updates = self.compute_weight_updates(weights)
        count = 0

        for (pre_id, post_id), delta_w in updates.items():
            if abs(delta_w) < 1e-10:
                continue
            if pre_id >= len(weights):
                continue

            current_w = weights[pre_id].get(post_id, 0.0)
            new_w = current_w + delta_w

            # クリッピング
            new_w = max(self.w_min, min(self.w_max, new_w))

            # 最小閾値以下のシナプスは刈り込み
            if new_w <= self.w_min + 1e-6:
                weights[pre_id].pop(post_id, None)
            else:
                weights[pre_id][post_id] = new_w

            count += 1

        # ホメオスタシス: 全ニューロンの重み合計を目標値にスケーリング
        if self.homeostatic_target is not None:
            self._apply_homeostasis(weights)

        self.total_updates += count
        return count

    def _apply_homeostasis(self, weights: List[Dict[int, float]]) -> None:
        """恒常性シナプススケーリングを適用する。"""
        for pre_id in range(len(weights)):
            if not weights[pre_id]:
                continue
            total_w = sum(weights[pre_id].values())
            if total_w <= 0:
                continue
            assert self.homeostatic_target is not None
            ratio = self.homeostatic_target / total_w
            # 急激な変化を防ぐため、ゆっくり目標に近づける
            scale = 1.0 + self.homeostatic_rate * (ratio - 1.0)
            scale = max(0.95, min(1.05, scale))
            for post_id in weights[pre_id]:
                weights[pre_id][post_id] = max(
                    self.w_min,
                    min(self.w_max, weights[pre_id][post_id] * scale),
                )

    def get_stats(self) -> Dict[str, float]:
        """学習統計情報を返す。"""
        return {
            "total_updates": float(self.total_updates),
            "cumulative_reward": self.cumulative_reward,
            "reward_baseline": self.dopamine.reward_baseline,
            "dopamine_signal": self.dopamine.get_signal(),
            "last_rpe": self.dopamine.last_rpe,
            "active_ltp_traces": float(len(self.eligibility.ltp_traces)),
            "active_ltd_traces": float(len(self.eligibility.ltd_traces)),
        }

    def reset(self) -> None:
        """全内部状態をリセットする。"""
        self.dopamine.reset()
        self.eligibility.reset()
        self.total_updates = 0
        self.cumulative_reward = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dopamine": self.dopamine.state_dict(),
            "eligibility": self.eligibility.state_dict(),
            "total_updates": self.total_updates,
            "cumulative_reward": self.cumulative_reward,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "dopamine" in state:
            self.dopamine.load_state_dict(state["dopamine"])
        if "eligibility" in state:
            self.eligibility.load_state_dict(state["eligibility"])
        self.total_updates = int(state.get("total_updates", 0))
        self.cumulative_reward = float(state.get("cumulative_reward", 0.0))
