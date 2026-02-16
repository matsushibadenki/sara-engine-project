_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/learning/stdp.py",
    "//": "タイトル: STDP (Spike-Timing-Dependent Plasticity) による事前学習",
    "//": "目的: 誤差逆伝播法や行列演算を用いず、SNNの生物学的学習則(STDP)を用いて教師なしで言語の共起関係や文法構造を事前学習する。"
}

import math
from typing import List, Any, Tuple

class STDPPretrainer:
    """
    STDP (Spike-Timing-Dependent Plasticity) を用いた教師なし自己回帰学習モジュール。
    スパイクの発火タイミングの前後関係に基づいてシナプス重み（SaraGPTのsynapses）を更新する。
    行列演算や誤差逆伝播法は一切使用せず、純粋な局所的アップデートのみで文脈の因果関係を獲得する。
    """
    def __init__(self, window_size: int = 4, a_plus: float = 1.0, a_minus: float = 0.5, tau: float = 2.0, w_max: float = 15.0):
        """
        Args:
            window_size: 過去何ステップ前までの発火をSTDPの対象とするか（時間窓）
            a_plus: LTP（長期増強）の最大変化量
            a_minus: LTD（長期抑圧）の最大変化量
            tau: 時間減衰の時定数（ステップ数が離れるほど効果が薄れる）
            w_max: シナプス重みの上限（Homeostasis: 発散防止）
        """
        self.window_size = window_size
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau = tau
        self.w_max = w_max

    def pretrain(self, model: Any, corpus: List[str]):
        """
        大量のコーパスを与えて教師なし事前学習を実行する。
        model は SaraGPT クラスのインスタンスを想定。
        """
        for i, text in enumerate(corpus):
            self._train_sequence(model, text)

    def _train_sequence(self, model: Any, text: str):
        """
        1つの文章（シーケンス）に対してSTDP則を適用し、シナプス結合を更新する。
        """
        token_ids = model.encoder.tokenizer.encode(text)
        if not token_ids:
            return

        # 終了状態の学習のために <eos> を付与
        eos_id = model.encoder.tokenizer.vocab.get("<eos>")
        if eos_id is not None and token_ids[-1] != eos_id:
            token_ids.append(eos_id)
            
        # 発火履歴バッファ: [(時刻t, SDRのリスト)]
        spike_history: List[Tuple[int, List[int]]] = []
        
        for t, tid in enumerate(token_ids):
            # 現在の時刻 t で発火したニューロン群（Post-synaptic）
            current_sdr = model.encoder._get_token_sdr(tid)
            
            # 過去の時間窓内の発火履歴との間でSTDPを計算
            for past_t, past_sdr in spike_history:
                dt = t - past_t
                if dt > self.window_size:
                    continue
                
                # 1. LTP (Long-Term Potentiation: 長期増強)
                dw_plus = self.a_plus * math.exp(-dt / self.tau)
                
                for pre in past_sdr:
                    if pre not in model.synapses:
                        model.synapses[pre] = {}
                    for post in current_sdr:
                        current_w = model.synapses[pre].get(post, 0.0)
                        new_w = current_w + dw_plus
                        model.synapses[pre][post] = min(new_w, self.w_max)
                        
                # 2. LTD (Long-Term Depression: 長期抑圧)
                dw_minus = self.a_minus * math.exp(-dt / self.tau)
                
                for pre in current_sdr:
                    if pre in model.synapses:
                        for post in past_sdr:
                            if post in model.synapses[pre]:
                                current_w = model.synapses[pre][post]
                                new_w = current_w - dw_minus
                                if new_w <= 0.0:
                                    del model.synapses[pre][post]
                                else:
                                    model.synapses[pre][post] = new_w
                                    
            # 現在の発火を履歴に追加
            spike_history.append((t, current_sdr))
            
            # バッファが時間窓を超えたら古い記憶を捨てる
            if len(spike_history) > self.window_size:
                spike_history.pop(0)