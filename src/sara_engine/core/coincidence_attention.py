_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/coincidence_attention.py",
    "//": "ファイルの日本語タイトル: 同期発火型アテンション",
    "//": "ファイルの目的や内容: 既存のSpikeSelfAttentionとは異なり、重みパラメータを持たず、純粋なSetの積集合(Coincidence)のみで情報の重要度をルーティングする概念実証用モジュール。"
}

from typing import List, Set

class SpikeDrivenAttention:
    """
    Spike-Driven Attention Mechanism based on pure coincidence detection.
    Replaces matrix multiplication and Softmax with set intersections.
    """
    def __init__(self, context_size: int = 128, threshold: float = 2.0):
        self.context_size = context_size
        self.threshold = threshold
        
        # 過去のKeyとValueのスパイク状態を保持するリングバッファ
        self.key_buffer: List[Set[int]] = []
        self.value_buffer: List[Set[int]] = []
        
        # 各コンテキスト位置のアテンション膜電位
        self.attention_potentials: List[float] = []

    def reset_state(self):
        self.key_buffer.clear()
        self.value_buffer.clear()
        self.attention_potentials.clear()

    def forward(self, q_spikes: Set[int], k_spikes: Set[int], v_spikes: Set[int]) -> Set[int]:
        """
        1ステップのスパイク処理。
        Q, K, Vはそれぞれ「発火したニューロンのインデックスの集合(Set)」として入力される。
        """
        # 1. 記憶の更新 (Memory Update)
        self.key_buffer.append(k_spikes)
        self.value_buffer.append(v_spikes)
        
        # コンテキストサイズを超えたら古い記憶から忘却
        if len(self.key_buffer) > self.context_size:
            self.key_buffer.pop(0)
            self.value_buffer.pop(0)

        # 2. 同期発火検出 (Coincidence Detection: QとKの照合)
        self.attention_potentials = [0.0] * len(self.key_buffer)
        for i, past_k in enumerate(self.key_buffer):
            # Queryスパイクと過去のKeyスパイクが一致(同期)したニューロン数
            coincidence = len(q_spikes.intersection(past_k))
            self.attention_potentials[i] = float(coincidence)

        # 3. 発火とルーティング (Fire & Route: Vの抽出)
        output_spikes = set()
        for i, potential in enumerate(self.attention_potentials):
            if potential >= self.threshold:
                output_spikes.update(self.value_buffer[i])
                
        return output_spikes