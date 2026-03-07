# {
#     "//": "ディレクトリパス: src/sara_engine/models/hierarchical_reservoir.py",
#     "//": "ファイルの日本語タイトル: 階層的Liquid Reservoir (同期制御版)",
#     "//": "ファイルの目的や内容: 階層ごとに異なる神経振動（リズム）を適用。上位層のTheta波が下位層のゲートを制御するCommunication Through Coherenceを実装し、情報の階層的な結合能力を強化。"
# }

from typing import List
from .liquid_reservoir import LiquidReservoir
from ..dynamics.oscillation import OscillationManager


class HierarchicalLiquidReservoir:
    """
    神経振動（Oscillation）によって層間の通信が同期制御される階層ネットワーク。
    """

    def __init__(self, size_per_layer: int = 500):
        self.n_neurons = size_per_layer
        self.global_step = 0
        self.current_time = 0.0
        self.oscillation_manager = OscillationManager()

        # 各層の初期化
        self.layer1 = LiquidReservoir(
            n_neurons=self.n_neurons, p_connect=0.1, dt=1.0)  # Gamma (Fast)
        self.layer2 = LiquidReservoir(
            n_neurons=self.n_neurons, p_connect=0.1, dt=1.0)  # Alpha (Medium)
        self.layer3 = LiquidReservoir(
            n_neurons=self.n_neurons, p_connect=0.1, dt=1.0)  # Theta (Slow)

    def step(self, external_currents: List[float]) -> List[int]:
        self.current_time += 1.0
        phases = self.oscillation_manager.get_phase_effects(self.current_time)

        # --- 各層の閾値を自身の脳波リズムで変動させる (Oscillatory Gating) ---

        # Layer 1: Gamma波（高速な特徴抽出）
        self.layer1.v_thresh = [30.0 + (phases["gamma"] * 5.0)
                                for _ in range(self.layer1.n)]
        l1_spikes = self.layer1.step(external_currents)

        # Layer 2: Alpha波（情報のフィルタリング）
        self.layer2.v_thresh = [30.0 + (phases["alpha"] * 5.0)
                                for _ in range(self.layer2.n)]
        l2_inputs = [
            1.0 if i in l1_spikes else 0.0 for i in range(self.layer2.n)]
        l2_spikes = self.layer2.step(l2_inputs)

        # Layer 3: Theta波（文脈の統合）
        self.layer3.v_thresh = [30.0 + (phases["theta"] * 5.0)
                                for _ in range(self.layer3.n)]
        l3_inputs = [
            1.0 if i in l2_spikes else 0.0 for i in range(self.layer3.n)]
        l3_spikes = self.layer3.step(l3_inputs)

        return l3_spikes
