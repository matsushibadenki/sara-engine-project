# Directory Path: src/sara_engine/models/spiking_jepa.py
# English Title: Spiking Joint Embedding Predictive Architecture
# Purpose/Content: Implementation of a backpropagation-free, pure list-based Spiking JEPA. Uses local STDP and energy minimization for self-supervised predictive coding in a sparse distributed representation (SDR) space. Multi-language support included.

import random

class EnergyMinimizer:
    """
    誤差逆伝播の代わりに、局所的なエネルギー（予測誤差）を最小化するモジュール。
    トップダウンの予測スパイクと、ボトムアップの観測スパイクを比較し、
    一致しない（Surpriseがある）場合のみSTDPの学習シグナル（Reward/Penalty）を生成する。
    """
    def __init__(self, size: int):
        self.size = size

    def compute_surprise_signal(self, predicted_spikes: list[int], target_spikes: list[int]) -> tuple[list[int], float]:
        """
        予測と目標の差異を計算し、学習の強度（グローバルな報酬/ペナルティシグナル）と
        驚き（Surprise）のスパイクパターンを返す。
        """
        surprise_spikes = [0] * self.size
        error_count = 0
        match_count = 0

        for i in range(self.size):
            p = predicted_spikes[i]
            t = target_spikes[i]
            if p != t:
                surprise_spikes[i] = 1
                error_count += 1
            elif p == 1 and t == 1:
                match_count += 1
        
        # 予測が一致すれば報酬（LTPを促進）、外れればペナルティ（LTDを促進）
        total_active = sum(target_spikes)
        if total_active == 0:
            return surprise_spikes, 0.0 # ターゲットが発火していない場合は学習しない
            
        accuracy = match_count / total_active
        
        # 予測精度が高いほどポジティブな学習信号、低いほどネガティブな信号
        learning_signal = (accuracy * 2.0) - 1.0 
        
        return surprise_spikes, learning_signal


class SpikingJEPA:
    """
    Spiking Hierarchical Joint Embedding Predictive Architecture.
    
    ピクセルレベルの再構成ではなく、意味的・抽象的なSDR（スパース分散表現）空間での
    「未来の状態予測」を自己教師あり学習（STDPベース）で獲得するモジュール。
    行列演算・誤差逆伝播は一切使用しない。
    """
    def __init__(self, context_size: int, target_size: int, hidden_size: int):
        self.context_size = context_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        
        # コンテキスト（現在の状態）から潜在表現を生成するプロジェクタ
        self.context_projector = self._init_sparse_weights(context_size, hidden_size)
        
        # 潜在表現から未来のターゲット状態を予測するプレディクタ
        self.predictor = self._init_sparse_weights(hidden_size, target_size)
        
        self.energy_minimizer = EnergyMinimizer(target_size)
        
        # 学習パラメータ (STDP & BCM風の恒常性)
        self.learning_rate = 0.1
        self.w_max = 5.0
        self.prune_threshold = 0.01
        
        # 膜電位バッファ
        self.hidden_potentials = [0.0] * hidden_size
        self.target_potentials = [0.0] * target_size
        self.threshold = 0.5

    def get_status_message(self, lang: str = "en") -> str:
        """多言語ステータス取得機能"""
        messages = {
            "en": "Spiking JEPA initialized. Backpropagation-free predictive coding is active.",
            "ja": "Spiking JEPAが初期化されました。誤差逆伝播を使わない予測符号化が有効です。",
            "fr": "Spiking JEPA initialisé. Le codage prédictif sans rétropropagation est actif."
        }
        return messages.get(lang, messages["en"])

    def _init_sparse_weights(self, in_size: int, out_size: int) -> list[dict[int, float]]:
        """純粋なリストと辞書でスパースなシナプス結合を初期化"""
        synapses = []
        for _ in range(out_size):
            connections = {}
            for i in range(in_size):
                if random.random() < 0.3: # スパース結線
                    connections[i] = random.uniform(0.1, 0.5)
            synapses.append(connections)
        return synapses

    def _forward_layer(self, input_spikes: list[int], synapses: list[dict[int, float]], potentials: list[float]) -> list[int]:
        """発火のフォワードプロパゲーション"""
        out_spikes = [0] * len(potentials)
        active_inputs = [i for i, s in enumerate(input_spikes) if s == 1]
        
        for j in range(len(potentials)):
            potentials[j] *= 0.8 # 漏れ(Leak)
            for i in active_inputs:
                if i in synapses[j]:
                    potentials[j] += synapses[j][i]
                    
            if potentials[j] >= self.threshold:
                out_spikes[j] = 1
                potentials[j] = 0.0 # 発火後リセット
                
        return out_spikes

    def _update_weights_stdp(self, input_spikes: list[int], output_spikes: list[int], synapses: list[dict[int, float]], signal: float):
        """
        局所的な学習信号に基づくHebbian / Anti-Hebbian学習 (バックプロパゲーション代替)
        """
        active_inputs = set([i for i, s in enumerate(input_spikes) if s == 1])
        
        for j, out_fired in enumerate(output_spikes):
            if not out_fired:
                continue
                
            current_synapses = synapses[j]
            for i in list(current_synapses.keys()):
                # 入力と出力が同時に発火している場合 (因果的)
                if i in active_inputs:
                    # シグナルが正ならLTP、負ならLTD
                    delta = self.learning_rate * signal * (self.w_max - current_synapses[i])
                    current_synapses[i] += delta
                else:
                    # 入力がないのに出力が発火した場合 (ヘテロシナプティックLTD)
                    current_synapses[i] -= (self.learning_rate * 0.1 * current_synapses[i])
                
                # 刈り込み (Pruning)
                if current_synapses[i] < self.prune_threshold:
                    del current_synapses[i]
                elif current_synapses[i] > self.w_max:
                    current_synapses[i] = self.w_max

    def step(self, context_spikes: list[int], target_spikes: list[int], learning: bool = True) -> tuple[list[int], float]:
        """
        1回のタイムステップの推論と学習を実行する。
        
        1. コンテキストから潜在表現(hidden)を生成
        2. 潜在表現から未来のターゲット状態を予測(predicted)
        3. 実際のターゲット状態(target)と予測を比較し、エネルギー(誤差)を計算
        4. エネルギーを最小化するように局所的にシナプス重みを更新
        """
        
        # 1. コンテキストのエンコーディング
        hidden_spikes = self._forward_layer(context_spikes, self.context_projector, self.hidden_potentials)
        
        # 2. 未来状態の予測
        predicted_spikes = self._forward_layer(hidden_spikes, self.predictor, self.target_potentials)
        
        # 3. エネルギー(Surprise)の計算
        surprise_spikes, learning_signal = self.energy_minimizer.compute_surprise_signal(predicted_spikes, target_spikes)
        
        # 4. 局所的STDPによる自己教師あり学習
        if learning and sum(target_spikes) > 0:
            # プレディクタの学習 (潜在表現 -> ターゲット)
            self._update_weights_stdp(hidden_spikes, predicted_spikes, self.predictor, learning_signal)
            
            # プロジェクタの学習 (コンテキスト -> 潜在表現)
            # ※本来のJEPAではターゲットネットワークはEMA(指数移動平均)で更新するが、
            # SNNでは学習シグナルを弱めてローカルに更新することで模倣する
            self._update_weights_stdp(context_spikes, hidden_spikes, self.context_projector, learning_signal * 0.5)
            
        # 今後の拡張(能動的推論)に向け、Surpriseパターンを返す
        return surprise_spikes, learning_signal