_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/hal.py",
    "//": "ファイルの日本語タイトル: ハードウェア抽象化層 (HAL)",
    "//": "ファイルの目的や内容: Pythonリファレンス、Rustマルチコア、および将来のニューロモルフィック専用ハードウェアを透過的に切り替えてスパイク伝播を実行するためのインターフェース。"
}

import logging
from typing import List, Dict

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
        self.weights = []
        
    def set_weights(self, weights: List[Dict[int, float]]):
        self.weights = weights
        
    def propagate(self, active_spikes: List[int], threshold: float, max_out: int) -> List[int]:
        potentials = {}
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
            from sara_engine import sara_rust_core
            self.engine = sara_rust_core.SpikeEngine()
            self.available = True
        except ImportError:
            self.available = False
            logging.warning("Rust core not found. Please compile with maturin.")
            
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
        
    def set_weights(self, weights: List[Dict[int, float]]):
        # シミュレーション：重みの量子化とオンチップメモリ・クロスバーアレイへのマッピング
        self.weights_mapped = True
        self.cores_used = min(128, len(weights) // 100 + 1)
        
    def propagate(self, active_spikes: List[int], threshold: float, max_out: int) -> List[int]:
        if not self.weights_mapped:
            return []
        # モックとしてダミーの振る舞い（ここではPython実装を内部で借用して返す）
        # ※本来はPCIe等を経由した非同期通信が行われる
        return active_spikes[:max_out] 
        
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