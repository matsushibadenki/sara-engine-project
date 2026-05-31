from .layers import SpikeFeedForward
from typing import List, Dict
import random
from ..learning.homeostasis import AdaptiveThresholdHomeostasis
# ディレクトリパス: src/sara_engine/core/cortical_columns.py
# ファイルの日本語タイトル: スパイキング・大脳皮質カラム (MoE代替)
# ファイルの目的や内容: TransformersのMoE (Mixture of Experts) の生物学的代替。側抑制(Lateral Inhibition)とホメオスタシス(発火頻度適応)によるWinner-Take-All(WTA)回路を用いて、特定エキスパートの独占を防ぎながら入力を最適にルーティングする。
try:
    import sara_rust_core
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False


class SpikingCorticalColumns:
    """
    Biological alternative to Transformer's Mixture of Experts (MoE).
    Uses Spike-Timing Dependent routing, WTA (Winner-Take-All) competition,
    and Homeostasis (Spike Frequency Adaptation) to prevent rich-club dominance.
    """

    def __init__(self, embed_dim: int, num_experts: int = 8, top_k: int = 2, density: float = 0.1):
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_rust = HAS_RUST_CORE

        # Each expert acts as a cortical minicolumn (equivalent to a FeedForward Network)
        self.experts = [SpikeFeedForward(
            embed_dim, embed_dim * 2, density) for _ in range(num_experts)]
        self.step_counter = 0

        initial_weights: List[Dict[int, float]] = [{}
                                                   for _ in range(embed_dim)]
        for i in range(embed_dim):
            # Initially connect to a random subset of experts
            targets = random.sample(
                range(num_experts), max(1, int(num_experts * 0.5)))
            for t in targets:
                initial_weights[i][t] = random.uniform(0.1, 1.0)

        if self.use_rust:
            self.router = sara_rust_core.SpikeWTARouter(
                embed_dim, num_experts, self.top_k)
            self.router.set_weights(initial_weights)
        else:
            self.router_weights = initial_weights
            self.router_homeostasis = AdaptiveThresholdHomeostasis(
                target_rate=max(0.05, self.top_k / max(1, num_experts)),
                adaptation_rate=0.4,
                decay=0.92,
                min_threshold=0.0,
                max_threshold=4.0,
                global_weight=0.3,
            )

    def reset_state(self):
        for expert in self.experts:
            if hasattr(expert, 'reset_state'):
                expert.reset_state()
        self.step_counter = 0
        if not self.use_rust:
            self.router_homeostasis.reset()

    def state_dict(self) -> Dict:
        return {
            "embed_dim": self.embed_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "experts": [expert.state_dict() for expert in self.experts],
            "router_weights": self.router.get_weights() if self.use_rust else self.router_weights,
            "router_thresholds": self.router.get_thresholds() if self.use_rust else self.router_homeostasis.thresholds,
            "router_homeostasis": None if self.use_rust else self.router_homeostasis.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.embed_dim = state["embed_dim"]
        self.num_experts = state["num_experts"]
        self.top_k = state["top_k"]

        for expert, exp_state in zip(self.experts, state["experts"]):
            expert.load_state_dict(exp_state)

        if self.use_rust:
            self.router.set_weights(state["router_weights"])
            if "router_thresholds" in state:
                self.router.set_thresholds(state["router_thresholds"])
        else:
            self.router_weights = state["router_weights"]
            if "router_homeostasis" in state:
                self.router_homeostasis.load_state_dict(state["router_homeostasis"])
            else:
                self.router_homeostasis.thresholds = {
                    int(k): float(v)
                    for k, v in state.get("router_thresholds", {}).items()
                }

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        if self.use_rust:
            active_experts = self.router.route(x_spikes, learning)
        else:
            # Python fallback WTA routing with Homeostasis
            expert_potentials = {i: 0.0 for i in range(self.num_experts)}
            for spike in x_spikes:
                if spike < self.embed_dim:
                    for exp_id, weight in self.router_weights[spike].items():
                        expert_potentials[exp_id] += weight

            # Apply Homeostasis (subtract fatigue threshold)
            adjusted_potentials = {
                i: expert_potentials[i] - self.router_homeostasis.get_threshold(i, 0.0)
                for i in range(self.num_experts)
            }
            sorted_experts = sorted(
                adjusted_potentials.items(), key=lambda x: x[1], reverse=True)
            active_experts = [exp_id for exp_id,
                              pot in sorted_experts[:self.top_k]]

            if learning:
                self.router_homeostasis.update(
                    active_experts, population_size=self.num_experts)
            else:
                self.router_homeostasis.update(
                    [], population_size=self.num_experts)

        out_spikes = set()
        for exp_id in active_experts:
            expert_out = self.experts[exp_id].forward(
                x_spikes, learning=learning)
            out_spikes.update(expert_out)

        if learning and active_experts:
            self.step_counter += 1
            if self.use_rust:
                self.router.update_weights(x_spikes, active_experts, 0.05)
                if self.step_counter % 100 == 0:
                    self.router.decay_weights(0.99)
            else:
                winner_set = set(active_experts)
                for spike in x_spikes:
                    if spike < self.embed_dim:
                        for exp_id in list(self.router_weights[spike].keys()):
                            if exp_id in winner_set:
                                self.router_weights[spike][exp_id] = min(
                                    3.0, self.router_weights[spike][exp_id] + 0.05)
                            else:
                                self.router_weights[spike][exp_id] = max(
                                    0.0, self.router_weights[spike][exp_id] - 0.01)
                                if self.router_weights[spike][exp_id] < 0.05:
                                    del self.router_weights[spike][exp_id]
                        for exp_id in winner_set:
                            if exp_id not in self.router_weights[spike]:
                                self.router_weights[spike][exp_id] = 0.1

                if self.step_counter % 100 == 0:
                    for w_dict in self.router_weights:
                        for k in list(w_dict.keys()):
                            w_dict[k] *= 0.99
                            if w_dict[k] < 0.05:
                                del w_dict[k]

        return list(out_spikes)
