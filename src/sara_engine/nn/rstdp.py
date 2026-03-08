# ディレクトリパス: src/sara_engine/nn/rstdp.py
# ファイルの日本語タイトル: 報酬変調STDP層 (R-STDP)
# ファイルの目的や内容:
#   誤差逆伝播法(BP)に依存せず、適格度トレースと遅延報酬を用いた「3要素学習則」によって
#   大域的最適化を行う強化学習用SNNモジュール。
#   STDP窓関数、ε-greedy探索ポリシー、ドーパミン信号モデルによるRPEベース学習、
#   恒常性維持を統合した本格的なR-STDP層。

import math
import random
from typing import List, Dict, Tuple
from .module import SNNModule


class RewardModulatedLinearSpike(SNNModule):
    """R-STDP (Reward-Modulated STDP) に基づく線形スパイク層。

    3要素学習則を採用し、
    1. Pre-synaptic (シナプス前発火)
    2. Post-synaptic (シナプス後発火)
    3. Neuromodulator (ドーパミンなどの遅延報酬シグナル)
    の組み合わせにより、BPなしで強化学習を実現する。

    拡張機能:
        - STDP窓関数 (指数関数型) による精密なトレース計算
        - ε-greedy 探索ポリシーの組み込み
        - ドーパミン信号モデル (RPE: 報酬予測誤差) による適応的学習
        - 発火率ベースのホメオスタシス (恒常性維持)
        - Soft-bound制約による重みの安定化
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 0.5,
        threshold: float = 0.5,
        w_max: float = 3.0,
        tau_stdp_plus: float = 20.0,
        tau_stdp_minus: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 0.5,
        trace_decay: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        baseline_decay: float = 0.99,
        homeostatic_target_rate: float = 0.2,
        homeostatic_strength: float = 0.01,
    ) -> None:
        """
        Args:
            in_features: 入力ニューロン数。
            out_features: 出力ニューロン数。
            density: 初期結線密度 (0〜1)。
            threshold: 発火閾値。
            w_max: シナプス重みの最大値。
            tau_stdp_plus: STDP窓の LTP 時定数。
            tau_stdp_minus: STDP窓の LTD 時定数。
            a_plus: LTP の振幅。
            a_minus: LTD の振幅。
            trace_decay: 適格度トレースの時間減衰率。
            epsilon: ε-greedy 探索の初期ε値。
            epsilon_decay: ε の指数減衰率 (エピソードごと)。
            epsilon_min: ε の最小値。
            baseline_decay: 報酬ベースラインの EMA 減衰率。
            homeostatic_target_rate: ホメオスタシスの目標発火率。
            homeostatic_strength: ホメオスタシスの調整強度。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.w_max = w_max
        self.tau_stdp_plus = tau_stdp_plus
        self.tau_stdp_minus = tau_stdp_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.trace_decay = trace_decay

        # ε-greedy 探索パラメータ
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # ドーパミン信号 (報酬ベースライン追跡)
        self.baseline_decay = baseline_decay
        self.reward_baseline: float = 0.0
        self.reward_count: int = 0

        # ホメオスタシス
        self.homeostatic_target_rate = homeostatic_target_rate
        self.homeostatic_strength = homeostatic_strength
        self.firing_rate_ema: Dict[int, float] = {}

        # シナプス重みの初期化
        self.weights: List[Dict[int, float]] = [{} for _ in range(in_features)]
        for i in range(in_features):
            num_connections = max(1, int(out_features * density))
            targets = random.sample(range(out_features), min(
                num_connections, out_features))
            for t in targets:
                self.weights[i][t] = random.uniform(0.1, 1.0)

        self.register_state("weights")

        # 適格度トレース: LTP と LTD を分離管理
        # LTP: (pre, post) → trace (因果的: pre→post)
        self.ltp_traces: Dict[Tuple[int, int], float] = {}
        # LTD: (pre, post) → trace (逆因果的: post→pre)
        self.ltd_traces: Dict[Tuple[int, int], float] = {}

        # 各ニューロンの最終発火ステップ (STDP窓関数用)
        self._pre_spike_step: Dict[int, int] = {}
        self._post_spike_step: Dict[int, int] = {}
        self._current_step: int = 0

    def forward(
        self,
        spikes: List[int],
        learning: bool = False,
    ) -> List[int]:
        """スパイクを入力し、膜電位計算・発火判定・トレース記録を行う。

        Args:
            spikes: 入力スパイクのニューロンIDリスト。
            learning: True の場合、適格度トレースを記録する。

        Returns:
            発火した出力ニューロンのIDリスト。
        """
        self._current_step += 1

        # 学習時、過去のトレースを自然減衰させる
        if learning:
            self._decay_traces()

        # 膜電位の計算
        potentials = [0.0] * self.out_features
        for s in spikes:
            if s < self.in_features:
                for target, weight in self.weights[s].items():
                    if target < self.out_features:
                        potentials[target] += weight

        # ホメオスタシスによる閾値調整
        adjusted_thresholds = [self.threshold] * self.out_features
        for j in range(self.out_features):
            rate = self.firing_rate_ema.get(j, self.homeostatic_target_rate)
            # 発火率が高すぎる → 閾値を上げる、低すぎる → 下げる
            adjustment = self.homeostatic_strength * \
                (rate - self.homeostatic_target_rate)
            adjusted_thresholds[j] = max(0.1, self.threshold + adjustment)

        # 発火判定 (Top-K + 閾値)
        active_spikes = [
            (i, p)
            for i, p in enumerate(potentials)
            if p > adjusted_thresholds[i]
        ]
        active_spikes.sort(key=lambda x: x[1], reverse=True)
        max_spikes = max(1, int(self.out_features * 0.3))
        out_spikes = [i for i, _p in active_spikes[:max_spikes]]

        # ε-greedy 探索: 学習中はランダムな行動を一定確率で選択
        if learning and random.random() < self.epsilon:
            random_spike = random.randint(0, self.out_features - 1)
            if random_spike not in out_spikes:
                out_spikes.append(random_spike)

        # 発火率のEMA更新
        for j in range(self.out_features):
            fired = 1.0 if j in out_spikes else 0.0
            prev_rate = self.firing_rate_ema.get(
                j, self.homeostatic_target_rate)
            self.firing_rate_ema[j] = 0.99 * prev_rate + 0.01 * fired

        # STDP窓関数に基づく適格度トレースの記録
        if learning:
            self._record_stdp_traces(spikes, out_spikes)

        return out_spikes

    def _record_stdp_traces(
        self,
        pre_spikes: List[int],
        post_spikes: List[int],
    ) -> None:
        """STDP窓関数に基づいてLTP/LTDトレースを記録する。

        STDP窓関数:
            Δt = t_post - t_pre
            Δt > 0 → LTP: A+ × exp(-|Δt| / τ+)
            Δt < 0 → LTD: A- × exp(-|Δt| / τ-)
        """
        step = self._current_step

        # 因果的ペア: pre fired now, post fired now → 最大 LTP
        for pre in pre_spikes:
            if pre >= self.in_features:
                continue
            for post in post_spikes:
                key = (pre, post)
                current = self.ltp_traces.get(key, 0.0)
                self.ltp_traces[key] = current + self.a_plus

        # 因果的ペア: pre fired earlier, post fired now → 減衰した LTP
        for post in post_spikes:
            for pre, pre_step in self._pre_spike_step.items():
                if pre in pre_spikes:
                    continue  # 同時発火は上で処理済み
                delta_t = step - pre_step
                if 0 < delta_t <= int(self.tau_stdp_plus * 3):
                    key = (pre, post)
                    ltp_amount = self.a_plus * \
                        math.exp(-delta_t / self.tau_stdp_plus)
                    current = self.ltp_traces.get(key, 0.0)
                    self.ltp_traces[key] = current + ltp_amount

        # 逆因果的ペア: post fired earlier, pre fired now → LTD
        for pre in pre_spikes:
            if pre >= self.in_features:
                continue
            for post, post_step in self._post_spike_step.items():
                if post in post_spikes:
                    continue  # 同時発火はLTPで処理
                delta_t = step - post_step
                if 0 < delta_t <= int(self.tau_stdp_minus * 3):
                    key = (pre, post)
                    ltd_amount = self.a_minus * \
                        math.exp(-delta_t / self.tau_stdp_minus)
                    current = self.ltd_traces.get(key, 0.0)
                    self.ltd_traces[key] = current + ltd_amount

        # 発火時刻の記録を更新
        for pre in pre_spikes:
            if pre < self.in_features:
                self._pre_spike_step[pre] = step
        for post in post_spikes:
            self._post_spike_step[post] = step

    def _decay_traces(self) -> None:
        """全トレースに時間減衰を適用する。"""
        keys_to_remove: List[Tuple[int, int]] = []
        for key in self.ltp_traces:
            self.ltp_traces[key] *= self.trace_decay
            if self.ltp_traces[key] < 0.01:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.ltp_traces[key]

        keys_to_remove = []
        for key in self.ltd_traces:
            self.ltd_traces[key] *= self.trace_decay
            if self.ltd_traces[key] < 0.01:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.ltd_traces[key]

    def apply_reward(self, reward: float, learning_rate: float = 0.1) -> None:
        """環境から遅延報酬を受け取り、3要素学習則でシナプス荷重を更新する。

        3要素学習則: Δw = η × (e_ltp - e_ltd) × δ
        ここで δ = reward - baseline（報酬予測誤差）

        Soft-bound制約: 重みが上下限に近づくほど更新量が自動的に減衰する。

        Args:
            reward: 環境からの報酬スカラー値。
            learning_rate: 重み更新の学習率。
        """
        # 報酬予測誤差 (RPE) の計算
        rpe = reward - self.reward_baseline

        # 報酬ベースラインの更新 (EMA)
        self.reward_count += 1
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline
            + (1.0 - self.baseline_decay) * reward
        )

        # 全LTP/LTDトレースから正味のトレースを算出して重みを更新
        all_keys = set(self.ltp_traces.keys()) | set(self.ltd_traces.keys())
        keys_to_remove: List[Tuple[int, int]] = []

        for key in all_keys:
            pre, post = key
            ltp = self.ltp_traces.get(key, 0.0)
            ltd = self.ltd_traces.get(key, 0.0)
            net_trace = ltp - ltd

            if pre < self.in_features and post in self.weights[pre]:
                current_w = self.weights[pre][post]

                # Soft-bound 付き3要素学習則
                raw_delta = learning_rate * net_trace * rpe

                if raw_delta > 0:
                    soft_factor = (self.w_max - current_w) / self.w_max
                else:
                    soft_factor = current_w / self.w_max

                soft_factor = max(0.0, min(1.0, soft_factor))
                delta_w = raw_delta * soft_factor

                new_w = current_w + delta_w
                self.weights[pre][post] = max(0.0, min(self.w_max, new_w))

            keys_to_remove.append(key)

        # 報酬適用後のトレース消費
        for key in keys_to_remove:
            self.ltp_traces.pop(key, None)
            self.ltd_traces.pop(key, None)

        # ε の減衰（探索の削減）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def eligibility_traces(self) -> Dict[Tuple[int, int], float]:
        """後方互換性のための統合トレースビュー。"""
        all_keys = set(self.ltp_traces.keys()) | set(self.ltd_traces.keys())
        result: Dict[Tuple[int, int], float] = {}
        for key in all_keys:
            net = self.ltp_traces.get(key, 0.0) - self.ltd_traces.get(key, 0.0)
            if abs(net) > 1e-8:
                result[key] = net
        return result

    @eligibility_traces.setter
    def eligibility_traces(self, value: Dict[Tuple[int, int], float]) -> None:
        """後方互換性のためのセッター。正の値はLTP、負の値はLTDに振り分ける。"""
        self.ltp_traces.clear()
        self.ltd_traces.clear()
        for key, val in value.items():
            if val >= 0:
                self.ltp_traces[key] = val
            else:
                self.ltd_traces[key] = abs(val)

    def get_stats(self) -> Dict[str, float]:
        """学習統計情報を返す。"""
        avg_rate = 0.0
        if self.firing_rate_ema:
            avg_rate = sum(self.firing_rate_ema.values()) / \
                len(self.firing_rate_ema)
        return {
            "epsilon": self.epsilon,
            "reward_baseline": self.reward_baseline,
            "reward_count": float(self.reward_count),
            "active_ltp_traces": float(len(self.ltp_traces)),
            "active_ltd_traces": float(len(self.ltd_traces)),
            "avg_firing_rate": avg_rate,
        }

    def reset_state(self) -> None:
        """動的状態をリセットする（重みは保持）。"""
        super().reset_state()
        self.ltp_traces.clear()
        self.ltd_traces.clear()
        self._pre_spike_step.clear()
        self._post_spike_step.clear()
        self._current_step = 0
        self.firing_rate_ema.clear()
