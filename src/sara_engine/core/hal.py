# [配置するディレクトリのパス]: ./src/sara_engine/core/hal.py
# [ファイルの日本語タイトル]: ハードウェア抽象化層 (HAL)
# [ファイルの目的や内容]: Pythonリファレンス、Rustマルチコア、および将来のニューロモルフィック専用ハードウェアを透過的に切り替えてスパイク伝播を実行するためのインターフェース。
from typing import List, Dict
import logging
# ディレクトリパス: src/sara_engine/core/hal.py
# ファイルの日本語タイトル: ハードウェア抽象化層 (HAL)
# ファイルの目的や内容: Pythonリファレンス、Rustマルチコア、および将来のニューロモルフィック専用ハードウェアを透過的に切り替えてスパイク伝播を実行するためのインターフェース。
class SpikeBackend:
    def set_weights(self, weights: List[Dict[int, float]]):
        raise NotImplementedError

    def propagate(self, active_spikes: List[int], threshold: float, max_out: int) -> List[int]:
        raise NotImplementedError

    def get_name(self) -> str:
        return "BaseBackend"


class PythonBackend(SpikeBackend):
    """ピュアPythonによるリファレンス実装。依存関係なしにどこでも動く。"""

    def __init__(self):
        self.weights: List[Dict[int, float]] = []

    def set_weights(self, weights: List[Dict[int, float]]):
        self.weights = weights

    def propagate(self, active_spikes: List[int], threshold: float, max_out: int) -> List[int]:
        potentials: Dict[int, float] = {}
        for pre in active_spikes:
            if pre < len(self.weights):
                for post, w in self.weights[pre].items():
                    potentials[post] = potentials.get(post, 0.0) + w

        active = [(post, p) for post, p in potentials.items() if p > threshold]
        active.sort(key=lambda x: x[1], reverse=True)
        return [post for post, _ in active[:max_out]]

    def get_name(self) -> str:
        return "Pure Python (Reference)"


class RustBackend(SpikeBackend):
    """Rayonを用いたマルチコアCPU最適化バックエンド。"""

    def __init__(self):
        try:
            from .. import sara_rust_core
            self.engine = sara_rust_core.SpikeEngine()
            self.available = True
        except ImportError:
            self.available = False
            logging.warning(
                "Rust core not found. Please compile with maturin.")

    def set_weights(self, weights: List[Dict[int, float]]):
        if self.available:
            self.engine.set_weights(weights)

    def propagate(self, active_spikes: List[int], threshold: float, max_out: int) -> List[int]:
        if self.available:
            return self.engine.propagate(active_spikes, threshold, max_out)
        return []

    def get_name(self) -> str:
        return "Rust Multi-core (Rayon Optimized)"


class MockNeuromorphicBackend(SpikeBackend):
    """
    将来の専用チップ（Intel Loihi, IBM TrueNorth等）への対応をシミュレートするモック。
    実際にはドライバを通じてチップへ重みを転送し、非同期にスパイクを送受信する。
    """

    def __init__(self):
        self.weights_mapped = False
        self.cores_used = 0
        self.quantized_weights: List[Dict[int, float]] = []
        self.last_event_cost = 0

    def set_weights(self, weights: List[Dict[int, float]]):
        # Simulate low-precision synapse mapping into bounded on-chip cores.
        self.weights_mapped = True
        self.cores_used = min(128, len(weights) // 100 + 1)
        self.quantized_weights = [
            {
                int(post): round(max(0.0, min(4.0, float(weight))) * 16.0) / 16.0
                for post, weight in row.items()
                if float(weight) > 0.0
            }
            for row in weights
        ]
        self.last_event_cost = 0

    def propagate(self, active_spikes: List[int], threshold: float, max_out: int) -> List[int]:
        if not self.weights_mapped:
            return []
        potentials: Dict[int, float] = {}
        event_cost = 0
        for pre in active_spikes:
            if 0 <= pre < len(self.quantized_weights):
                row = self.quantized_weights[pre]
                event_cost += len(row)
                for post, weight in row.items():
                    potentials[post] = potentials.get(post, 0.0) + weight
        self.last_event_cost = event_cost
        active = [(post, value) for post, value in potentials.items() if value > threshold]
        active.sort(key=lambda item: item[1], reverse=True)
        return [post for post, _value in active[:max(0, max_out)]]

    def mapping_report(self) -> Dict[str, float]:
        synapse_count = sum(len(row) for row in self.quantized_weights)
        return {
            "weights_mapped": float(self.weights_mapped),
            "cores_used": float(self.cores_used),
            "synapse_count": float(synapse_count),
            "last_event_cost": float(self.last_event_cost),
        }

    def get_name(self) -> str:
        return "Neuromorphic Hardware (Mock/Loihi Interface)"


class HardwareManager:
    """ユーザー指定に応じて最適なバックエンドを選択・管理する。"""

    def __init__(self, preferred: str = "rust"):
        self.backend: SpikeBackend = self._select_backend(preferred)

    def _select_backend(self, preferred: str) -> SpikeBackend:
        if preferred == "rust":
            backend = RustBackend()
            if backend.available:
                return backend
            logging.warning("Falling back to Python backend.")
            return PythonBackend()
        elif preferred == "chip":
            return MockNeuromorphicBackend()
        else:
            return PythonBackend()

    def get_backend(self) -> SpikeBackend:
        return self.backend
