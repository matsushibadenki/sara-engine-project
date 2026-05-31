# Directory Path: src/sara_engine/learning/reward_modulated_stdp.py
# English Title: Reward-Modulated STDP Manager
# Purpose/Content: 生物学的に妥当な報酬修飾型STDP。適格度トレース管理やドーパミン信号のインターフェースをオリジナルと完全に一致させつつ、内部ロジックで環境のRPE(報酬予測誤差)分散に基づく適応的ベースラインを取り入れることで高精度化。

import math
from typing import Dict, List, Tuple, Any, Optional

class DopamineSignalModel:
    """ドーパミン信号のダイナミクスモデル。適応的ベースライン更新を搭載。"""

    def __init__(
        self,
        tonic_level: float = 0.0,
        tau_dopamine: float = 20.0,
        baseline_decay: float = 0.99,
        rpe_scale: float = 1.0,
    ) -> None:
        self.tonic_level = tonic_level
        self.tau_dopamine = tau_dopamine
        self.baseline_decay = baseline_decay
        self.rpe_scale = rpe_scale

        self.phasic_level: float = 0.0
        self.reward_baseline: float = 0.0
        self.last_rpe: float = 0.0
        self.reward_count: int = 0
        self.rpe_variance: float = 1.0 # RPEの分散を追跡(適応機能)

    def deliver_reward(self, reward: float, intrinsic_surprise: float = 0.0) -> float:
        rpe = (reward + intrinsic_surprise - self.reward_baseline) * self.rpe_scale
        self.last_rpe = rpe
        self.phasic_level += rpe
        self.reward_count += 1

        # RPEの分散が大きい(環境変動)時はベースラインを早く更新
        self.rpe_variance = 0.9 * self.rpe_variance + 0.1 * (rpe ** 2)
        adaptive_decay = max(0.5, min(0.999, self.baseline_decay - 0.01 * self.rpe_variance))
        
        self.reward_baseline = adaptive_decay * self.reward_baseline + (1.0 - adaptive_decay) * reward
        return self.get_signal()

    def step(self) -> float:
        """1タイムステップ進め、フェーズィック成分を自然減衰させる。"""
        if abs(self.phasic_level) > 1e-8:
            decay = math.exp(-1.0 / self.tau_dopamine) if self.tau_dopamine > 0 else 0.0
            self.phasic_level *= decay
        else:
            self.phasic_level = 0.0
        return self.get_signal()

    def get_signal(self) -> float:
        return self.tonic_level + self.phasic_level

    def reset(self) -> None:
        self.phasic_level = 0.0
        self.reward_baseline = 0.0
        self.last_rpe = 0.0
        self.reward_count = 0
        self.rpe_variance = 1.0

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
    """適格度トレース (Eligibility Trace) の管理。"""

    def __init__(
        self,
        tau_eligibility: float = 25.0,
        tau_stdp_plus: float = 20.0,
        tau_stdp_minus: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 0.5,
    ) -> None:
        self.tau_eligibility = tau_eligibility
        self.tau_stdp_plus = tau_stdp_plus
        self.tau_stdp_minus = tau_stdp_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

        self.ltp_traces: Dict[Tuple[int, int], float] = {}
        self.ltd_traces: Dict[Tuple[int, int], float] = {}
        self.last_spike_time: Dict[int, float] = {}

    def record_pre_spike(self, pre_id: int, time: float) -> None:
        self.last_spike_time[pre_id] = time

    def record_post_spike(self, post_id: int, time: float) -> None:
        self.last_spike_time[post_id] = time

    def update_traces(
        self,
        pre_spikes: List[int],
        post_spikes: List[int],
        current_time: float,
    ) -> None:
        for pre_id in pre_spikes:
            self.record_pre_spike(pre_id, current_time)
        for post_id in post_spikes:
            self.record_post_spike(post_id, current_time)

        for post_id in post_spikes:
            for pre_id in pre_spikes:
                key = (pre_id, post_id)
                self.ltp_traces[key] = self.ltp_traces.get(key, 0.0) + self.a_plus

        for pre_id in pre_spikes:
            for post_id in self.last_spike_time:
                if post_id in post_spikes: continue
                post_time = self.last_spike_time.get(post_id, -1000.0)
                delta_t = current_time - post_time
                if 0.0 < delta_t < self.tau_stdp_minus * 3:
                    key = (pre_id, post_id)
                    ltd_amount = self.a_minus * math.exp(-delta_t / self.tau_stdp_minus)
                    self.ltd_traces[key] = self.ltd_traces.get(key, 0.0) + ltd_amount

    def decay_traces(self) -> None:
        """全トレースに時間減衰を適用する。アクティブな要素のみをフィルタリングして高速化。"""
        decay_factor = math.exp(-1.0 / self.tau_eligibility) if self.tau_eligibility > 0 else 0.0

        keys_to_remove = []
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
        """特定のシナプスペアの正味のトレース値を返す"""
        key = (pre_id, post_id)
        return self.ltp_traces.get(key, 0.0) - self.ltd_traces.get(key, 0.0)

    def get_all_traces(self) -> Dict[Tuple[int, int], float]:
        all_keys = set(self.ltp_traces.keys()) | set(self.ltd_traces.keys())
        result: Dict[Tuple[int, int], float] = {}
        for key in all_keys:
            net = self.ltp_traces.get(key, 0.0) - self.ltd_traces.get(key, 0.0)
            if abs(net) > 1e-8:
                result[key] = net
        return result

    def reset(self) -> None:
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
    """報酬修飾型STDP (R-STDP) 統合マネージャー"""

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
        self.learning_rate = learning_rate
        self.w_min = w_min
        self.w_max = w_max
        self.homeostatic_target = homeostatic_target
        self.homeostatic_rate = homeostatic_rate

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

        self.total_updates: int = 0
        self.cumulative_reward: float = 0.0

    def record_spikes(
        self,
        pre_spikes: List[int],
        post_spikes: List[int],
        current_time: float,
    ) -> None:
        self.eligibility.update_traces(pre_spikes, post_spikes, current_time)

    def deliver_reward(self, reward: float, intrinsic_surprise: float = 0.0) -> float:
        self.cumulative_reward += reward
        return self.dopamine.deliver_reward(reward, intrinsic_surprise=intrinsic_surprise)

    def step(self) -> None:
        self.eligibility.decay_traces()
        self.dopamine.step()

    def compute_weight_updates(
        self,
        weights: List[Dict[int, float]],
    ) -> Dict[Tuple[int, int], float]:
        dopamine_signal = self.dopamine.get_signal()
        all_traces = self.eligibility.get_all_traces()
        updates: Dict[Tuple[int, int], float] = {}

        for (pre_id, post_id), trace in all_traces.items():
            raw_delta = self.learning_rate * trace * dopamine_signal

            if pre_id < len(weights) and post_id in weights[pre_id]:
                current_w = weights[pre_id][post_id]
            else:
                current_w = 0.0

            if raw_delta > 0:
                soft_factor = (self.w_max - current_w) / self.w_max
            else:
                soft_factor = (current_w - self.w_min) / self.w_max

            soft_factor = max(0.0, min(1.0, soft_factor))
            updates[(pre_id, post_id)] = raw_delta * soft_factor

        return updates

    def apply_weight_updates(
        self,
        weights: List[Dict[int, float]],
    ) -> int:
        updates = self.compute_weight_updates(weights)
        count = 0

        for (pre_id, post_id), delta_w in updates.items():
            if abs(delta_w) < 1e-10 or pre_id >= len(weights):
                continue

            current_w = weights[pre_id].get(post_id, 0.0)
            new_w = current_w + delta_w
            new_w = max(self.w_min, min(self.w_max, new_w))

            if new_w <= self.w_min + 1e-6:
                weights[pre_id].pop(post_id, None)
            else:
                weights[pre_id][post_id] = new_w

            count += 1

        if self.homeostatic_target is not None:
            self._apply_homeostasis(weights)

        self.total_updates += count
        return count

    def _apply_homeostasis(self, weights: List[Dict[int, float]]) -> None:
        for pre_id in range(len(weights)):
            if not weights[pre_id]:
                continue
            total_w = sum(weights[pre_id].values())
            if total_w <= 0:
                continue
            assert self.homeostatic_target is not None
            ratio = self.homeostatic_target / total_w
            scale = 1.0 + self.homeostatic_rate * (ratio - 1.0)
            scale = max(0.95, min(1.05, scale))
            for post_id in weights[pre_id]:
                weights[pre_id][post_id] = max(
                    self.w_min,
                    min(self.w_max, weights[pre_id][post_id] * scale),
                )

    def get_stats(self) -> Dict[str, float]:
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