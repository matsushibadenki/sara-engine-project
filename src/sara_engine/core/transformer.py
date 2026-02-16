{
    "//": "ディレクトリパス: src/sara_engine/core/transformer.py",
    "//": "タイトル: Spike Transformer モデル (行列演算・誤差逆伝播完全排除版)",
    "//": "目的: Attention層(時間)とSTDP層(空間)を統合し、Transformerの代替となる生体模倣モデルを提供する。既存インターフェース(fit/predict等)と互換性を保ち、他スクリプトの破壊を防ぎつつNumPy依存を排除する。"
}

import json
import random
from typing import List, Dict

# 先ほど作成した行列演算不使用のSNNコンポーネントをインポート
from sara_engine.core.spike_attention import SpikeAttention
from sara_engine.learning.stdp import STDPLayer

class SpikePositionalEncoding:
    """
    NumPyの行列演算を使用せず、標準ライブラリのrandomを用いて
    疎（スパース）な位置エンコーディング用スパイクパターンを生成する。
    """
    def __init__(self, d_model: int, max_len: int = 1000, density: float = 0.1):
        self.d_model = d_model
        self.max_len = max_len
        self.pos_spikes = []
        
        # 再現性のためのシード固定（既存コードとの互換性）
        rng = random.Random(42)
        num_active = max(1, int(d_model * density))
        
        for _ in range(max_len):
            spikes = [0] * d_model
            # d_model個のニューロンから発火させるインデックスを重複なしで選択
            active_indices = rng.sample(range(d_model), num_active)
            for idx in active_indices:
                spikes[idx] = 1
            self.pos_spikes.append(spikes)

    def get_spikes(self, pos: int) -> List[int]:
        if pos < self.max_len:
            return self.pos_spikes[pos]
        return self.pos_spikes[-1]


class SpikeTransformerBlock:
    """
    1ステップのAttention(時間)とFFN/STDP(空間)の処理を統合したコアブロック。
    """
    def __init__(self, num_neurons: int, attn_decay: float = 0.8, attn_threshold: float = 1.5, ffn_threshold: float = 2.0):
        self.num_neurons = num_neurons
        self.attention = SpikeAttention(decay_rate=attn_decay, threshold=attn_threshold)
        self.ffn = STDPLayer(num_inputs=num_neurons, num_outputs=num_neurons, threshold=ffn_threshold)

    def process_step(self, input_spikes: List[int]) -> tuple[List[int], dict]:
        # 1. Self-Attention層
        attn_out, attn_scores = self.attention.process_step(
            query_spikes=input_spikes, 
            key_spikes=input_spikes, 
            value_spikes=input_spikes
        )

        # 2. 第1の残差接続（論理和による合流）
        res1_out = [
            1 if input_spikes[i] == 1 or attn_out[i] == 1 else 0 
            for i in range(self.num_neurons)
        ]

        # 3. FFN層（STDPによる自己組織化）
        ffn_out, ffn_potentials = self.ffn.process_step(res1_out)

        # 4. 第2の残差接続
        final_out = [
            1 if res1_out[i] == 1 or ffn_out[i] == 1 else 0 
            for i in range(self.num_neurons)
        ]

        debug_info = {
            "attn_scores": attn_scores,
            "ffn_potentials": ffn_potentials
        }
        return final_out, debug_info

    def reset(self):
        """ブロックの内部状態（発火履歴や膜電位）をリセットする"""
        self.attention.reset_state()
        # STDPの重み(weights)は長期記憶として保持するためリセットしないが、現在の興奮状態はクリアする
        self.ffn.potentials = [0.0] * self.ffn.num_outputs
        self.ffn.pre_traces = [0.0] * self.ffn.num_inputs
        self.ffn.post_traces = [0.0] * self.ffn.num_outputs


class SpikeTransformer:
    """
    外部スクリプト（run_transformer_task.py等）から呼び出されるメインインターフェース。
    既存のfit/predictメソッドを維持し、内部をSNNアーキテクチャで処理する。
    """
    def __init__(self, d_model: int, num_layers: int = 1, max_seq_len: int = 1000):
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoder = SpikePositionalEncoding(d_model=d_model, max_len=max_seq_len)
        # 複数のブロックを積み重ねる構造
        self.blocks = [SpikeTransformerBlock(num_neurons=d_model) for _ in range(num_layers)]

    def compute(self, x_ctx: List[int], pos: int, learning: bool = False) -> List[int]:
        """1ステップのフォワードパス（およびオンライン学習）を実行"""
        # 位置情報のスパイクを取得し、入力スパイクと合成（論理和）
        pos_spikes = self.pos_encoder.get_spikes(pos)
        current_spikes = [
            1 if x_ctx[i] == 1 or pos_spikes[i] == 1 else 0 
            for i in range(self.d_model)
        ]
        
        # 層（ブロック）を順番に通過させる
        for block in self.blocks:
            current_spikes, _ = block.process_step(current_spikes)
            
        return current_spikes

    def reset(self):
        """シーケンスごとの処理前にネットワークの状態を初期化"""
        for block in self.blocks:
            block.reset()

    def fit(self, sequence_spikes: List[List[int]], epochs: int = 200, verbose: bool = True):
        """教師データなしのオンライン自律学習を実行（互換性維持）"""
        if verbose:
            print(f"Training started for {epochs} epochs (Pure SNN Transformer Mode)...")
        for epoch in range(epochs):
            self.reset()
            for pos, spikes in enumerate(sequence_spikes):
                # process_step内でSTDPが自動的に学習を行うため、流し込むだけでよい
                self.compute(spikes, pos=pos, learning=True)
        if verbose:
            print("Training finished.")

    def predict(self, initial_spikes: List[int], steps: int = 3) -> List[List[int]]:
        """与えられた初期スパイクから先のシーケンスを推論して生成"""
        self.reset()
        current_spikes = initial_spikes
        generated = []
        for t in range(1, steps + 1):
            next_spikes = self.compute(current_spikes, pos=t, learning=False)
            generated.append(next_spikes)
            current_spikes = next_spikes
        return generated

    def save(self, filepath: str):
        """学習されたSTDPのシナプス結合荷重のみを保存"""
        state = {
            "d_model": self.d_model,
            "blocks_weights": [block.ffn.weights for block in self.blocks]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)

    def load(self, filepath: str):
        """保存されたSTDPのシナプス結合荷重を読み込む"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            self.d_model = state.get("d_model", self.d_model)
            weights = state.get("blocks_weights", [])
            for i, block in enumerate(self.blocks):
                if i < len(weights):
                    block.ffn.weights = weights[i]