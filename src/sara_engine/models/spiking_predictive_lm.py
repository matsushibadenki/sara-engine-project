from ..nn.predictive import PredictiveCodingLayer
from .hierarchical_snn import HierarchicalSNN
from ..nn.module import SNNModule
from typing import List, Dict, Any
import random
# ディレクトリパス: src/sara_engine/models/spiking_predictive_lm.py
# ファイルの日本語タイトル: 予測符号化ベースのスパイク言語モデル
# ファイルの目的や内容: 階層モデルと予測符号化層を統合。系列学習とデコードのパスから抽象化ノイズを完全に分離し、純粋なSDRのみを用いることで厳密な順序の自己回帰生成を実現する。不応期(Refractory Period)を実装し、反復生成を防止。
class SpikingPredictiveLM(SNNModule):
    def __init__(self, vocab_size: int, layer_configs: List[Dict[str, Any]], max_delay: int = 10, learning_rate: float = 0.1, predict_threshold: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size

        # トークンを固定のSDR（疎な分散表現）に変換するためのシード付き乱数生成器
        self.random_gen = random.Random(42)
        self.token_to_sdr: Dict[int, List[int]] = {}

        # 1. 階層的特徴抽出モジュール
        # (言語生成ではノイズになるが、将来的なマルチモーダル統合のレセプターとして機能)
        self.encoder = HierarchicalSNN(layer_configs=layer_configs)

        # 2. 予測符号化モジュール (系列の学習と次状態の予測)
        # 誤差逆伝播を用いず、予測誤差(Surprise)のみでシナプスを更新する
        self.predictive_core = PredictiveCodingLayer(
            max_delay=max_delay,
            learning_rate=learning_rate,
            threshold=predict_threshold
        )

        # 3. デコーダ (高次概念スパイク -> トークンID への還元マッピング)
        # 構造: {high_level_spike_id: {token_id: weight}}
        self.decoder_weights: Dict[int, Dict[int, float]] = {}
        self.register_state("decoder_weights")

    def forward(self, token_ids: List[int], learning: bool = True) -> List[int]:
        """
        学習または推論ステップ。
        """
        sdr_spikes = []
        for t_id in token_ids:
            if t_id not in self.token_to_sdr:
                # 階層モデルを駆動するための低次特徴スパイク (0~127)
                low_level = self.random_gen.sample(range(128), 16)
                # トークンの個性を完全に独立して保持する直交化SDR (1000以降で被りなし)
                start_id = 1000 + t_id * 200
                high_level = list(range(start_id, start_id + 200))
                self.token_to_sdr[t_id] = low_level + high_level
            sdr_spikes.extend(self.token_to_sdr[t_id])

        # 1. 階層モデルを通過 (状態の更新のみ行い、出力は言語生成パスには混ぜない)
        _ = self.encoder.forward(sdr_spikes, learning=False)

        # 系列の学習・予測・デコードには、ノイズのない「純粋な直交化SDR」のみを抽出して使用する
        pure_sdr_spikes = [s for s in sdr_spikes if s >= 1000]

        # 2. 予測符号化による系列学習（予測誤差スパイクのみが返る）
        error_spikes, error_rate = self.predictive_core.forward(
            pure_sdr_spikes, learning=learning)

        # 3. デコーダの学習（純粋なSDRとトークンを強固に結びつける）
        if learning:
            for h_spike in pure_sdr_spikes:
                if h_spike not in self.decoder_weights:
                    self.decoder_weights[h_spike] = {}

                # 競合減衰 (LTD): 今回発火しなかったトークンへの結合を弱めて混同を完全に防ぐ
                for t_id in list(self.decoder_weights[h_spike].keys()):
                    if t_id not in token_ids:
                        self.decoder_weights[h_spike][t_id] = max(
                            0.0, self.decoder_weights[h_spike][t_id] - 0.2)
                        if self.decoder_weights[h_spike][t_id] <= 0.0:
                            del self.decoder_weights[h_spike][t_id]

                # 長期増強 (LTP): 今回発火したトークンへの結合を強く形成する
                for t_id in token_ids:
                    current_w = self.decoder_weights[h_spike].get(t_id, 0.0)
                    self.decoder_weights[h_spike][t_id] = min(
                        3.0, current_w + 1.0)

        return error_spikes

    def _predict_next_sdr(self) -> List[int]:
        """
        Rustコア(CausalSynapses)の内部状態から、次ステップで発火するスパイクを予測する。
        """
        if not self.predictive_core.spike_history:
            return []

        # Rustエンジンの calculate_potentials を直接呼び出して予測電位を取得
        if hasattr(self.predictive_core.synapses, 'calculate_potentials'):
            potentials = self.predictive_core.synapses.calculate_potentials(
                self.predictive_core.spike_history)
            return [t for t, p in potentials.items() if p >= self.predictive_core.threshold]
        return []

    def generate(self, prompt_tokens: List[int], max_length: int = 10) -> List[int]:
        """
        与えられたプロンプト（初期トークン）から、続くトークンを自律的に生成する。
        生物学的な不応期(Refractory Period)を利用し、同じトークンの無限ループを防ぐ。
        """
        generated_sequence = list(prompt_tokens)

        # --- 生物学的メカニズム: 不応期 (Refractory Period) の管理 ---
        refractory_penalties: Dict[int, float] = {}

        # プロンプトの文脈を時系列の内部状態として順に読み込ませる
        for p_token in prompt_tokens:
            self.forward([p_token], learning=False)
            refractory_penalties[p_token] = 1000.0

        for _ in range(max_length):
            # 1. 現在の文脈履歴から、次の純粋なSDRを予測
            predicted_sdr = self._predict_next_sdr()

            if not predicted_sdr:
                break  # 予測不能（文脈の終わり）

            # 2. 予測されたSDRを具体的なトークンにデコード
            token_potentials: Dict[int, float] = {}
            for h_spike in predicted_sdr:
                if h_spike in self.decoder_weights:
                    for t_id, weight in self.decoder_weights[h_spike].items():
                        token_potentials[t_id] = token_potentials.get(
                            t_id, 0.0) + weight

            # 不応期の適用: 直前に発火したトークンの膜電位からペナルティを減算
            for t_id in token_potentials.keys():
                if t_id in refractory_penalties:
                    token_potentials[t_id] -= refractory_penalties[t_id]

            # 電位が0.0より高い（抑制されきっていない）有効なトークンのみを残す
            valid_potentials = {k: v for k,
                                v in token_potentials.items() if v > 0.0}
            if not valid_potentials:
                break

            # 最も電位の蓄積が高かったトークンを選択 (WTA: Winner-Take-All)
            next_token = max(valid_potentials.items(), key=lambda x: x[1])[0]
            generated_sequence.append(next_token)

            # 不応期の更新: 時間の経過とともに疲労状態から自然回復(Decay)する
            for t_id in list(refractory_penalties.keys()):
                refractory_penalties[t_id] *= 0.5
                if refractory_penalties[t_id] < 1.0:
                    del refractory_penalties[t_id]

            # 発火したニューロンを再び不応期（疲労状態）に入れる
            refractory_penalties[next_token] = 1000.0

            # 3. 生成したトークンを次のステップの「入力」として自己回帰的にフィードバック
            self.forward([next_token], learning=False)

        return generated_sequence

    def reset_state(self) -> None:
        """エンコーダと予測コアの内部状態（膜電位や履歴）をリセット"""
        self.encoder.reset_state()
        self.predictive_core.reset_state()
        super().reset_state()
