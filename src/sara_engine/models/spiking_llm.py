import math
import random
from typing import Any, Dict, List, Set, Tuple

from sara_engine.core.spike_attention import SpikeSelfAttention

_FILE_INFO = {
    "//1": "ディレクトリパス: src/sara_engine/models/spiking_llm.py",
    "//2": "ファイルの日本語タイトル: スパイキング・大規模言語モデルブロック（多層・STDP・恒常性可塑性対応）",
    "//3": "ファイルの目的や内容: 決定論的SDRテーブル + 直接コンテキストマッピングにより学習精度を大幅向上。"
           "Transformerの確率的ノイズを排除し、コンテキスト→次トークンを確実に記憶・再現する。",
}


class SpikingLayerNorm:
    def __init__(
        self,
        sdr_size: int,
        base_threshold: float = 1.0,
        target_active_ratio: float = 0.02,
    ):
        self.sdr_size = sdr_size
        self.base_threshold = base_threshold
        self.thresholds = [base_threshold] * sdr_size
        self.target_spikes = max(1, int(sdr_size * target_active_ratio))

    def forward(self, input_potentials: List[float]) -> List[int]:
        active_potentials = [(i, p)
                             for i, p in enumerate(input_potentials) if p > 0]

        if not active_potentials:
            for i in range(self.sdr_size):
                self.thresholds[i] = max(0.01, self.thresholds[i] - 0.02)
            return []

        active_ratio = len(active_potentials) / self.sdr_size
        avg_potential = sum(p for _, p in active_potentials) / \
            len(active_potentials)
        global_inhibition = avg_potential * active_ratio * 0.1

        spikes: List[int] = []
        for i, p in enumerate(input_potentials):
            effective_p = p - global_inhibition
            if effective_p >= self.thresholds[i]:
                spikes.append(i)

        max_allowed = self.target_spikes * 2
        min_required = max(1, int(self.target_spikes * 0.5))

        if len(spikes) > max_allowed:
            spikes.sort(key=lambda x: input_potentials[x], reverse=True)
            spikes = spikes[:max_allowed]
        elif len(spikes) < min_required and active_potentials:
            sorted_active = sorted(
                active_potentials, key=lambda x: x[1], reverse=True)
            for idx, _p in sorted_active:
                if len(spikes) >= min_required:
                    break
                if idx not in spikes:
                    spikes.append(idx)

        adjustment_rate = 0.01
        for i in range(self.sdr_size):
            if i in spikes:
                self.thresholds[i] += adjustment_rate
            else:
                self.thresholds[i] -= adjustment_rate * 0.8
            self.thresholds[i] = max(
                0.01, min(self.thresholds[i], self.base_threshold * 3.0))

        return sorted(spikes)


class STDP:
    def __init__(
        self,
        sdr_size: int,
        a_plus: float = 0.01,
        a_minus: float = 0.005,
        tau_plus: float = 5.0,
        tau_minus: float = 5.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
    ):
        self.sdr_size = sdr_size
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min
        self.reset_state()

    def reset_state(self) -> None:
        self.last_pre_times = [-1.0] * self.sdr_size
        self.last_post_times = [-1.0] * self.sdr_size

    def update_weights(
        self,
        t_step: int,
        pre_spikes: List[int],
        post_spikes: List[int],
        weights: List[Dict[int, float]],
    ) -> None:
        for pre_id in pre_spikes:
            self.last_pre_times[pre_id] = float(t_step)
        for post_id in post_spikes:
            self.last_post_times[post_id] = float(t_step)

        for post_id in post_spikes:
            t_post = self.last_post_times[post_id]
            for pre_id in range(self.sdr_size):
                if post_id in weights[pre_id]:
                    t_pre = self.last_pre_times[pre_id]
                    if t_pre >= 0:
                        delta_t = t_post - t_pre
                        if delta_t >= 0:
                            dw = self.a_plus * \
                                math.exp(-delta_t / self.tau_plus)
                        else:
                            dw = -self.a_minus * \
                                math.exp(delta_t / self.tau_minus)
                        new_w = weights[pre_id][post_id] + dw
                        weights[pre_id][post_id] = max(
                            self.w_min, min(self.w_max, new_w))


class SpikingTransformerBlock:
    def __init__(self, sdr_size: int, enable_learning: bool = True):
        self.sdr_size = sdr_size
        self.enable_learning = enable_learning
        self.attention = SpikeSelfAttention(embed_dim=sdr_size, density=0.05)

        self.layer_norm1 = SpikingLayerNorm(
            sdr_size, base_threshold=1.0, target_active_ratio=0.02)
        self.layer_norm2 = SpikingLayerNorm(
            sdr_size, base_threshold=1.2, target_active_ratio=0.02)

        self.ffn_w: List[Dict[int, float]] = [{} for _ in range(sdr_size)]
        self._init_sparse_weights(self.ffn_w, density=0.1)

        if self.enable_learning:
            self.stdp = STDP(sdr_size)

    def reset_state(self) -> None:
        if hasattr(self.attention, "reset_state"):
            self.attention.reset_state()
        if self.enable_learning:
            self.stdp.reset_state()

    def _init_sparse_weights(self, weights: List[Dict[int, float]], density: float) -> None:
        for i in range(self.sdr_size):
            num_connections = int(self.sdr_size * density)
            targets = random.sample(range(self.sdr_size), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.1, 0.5)

    def forward(self, input_spikes: List[int], t_step: int = 0) -> List[int]:
        att_spikes = self.attention.forward(
            input_spikes, learning=self.enable_learning)

        res_potentials_1 = [0.0] * self.sdr_size
        for s in set(input_spikes).union(set(att_spikes)):
            res_potentials_1[s] += 1.0
        norm1_spikes = self.layer_norm1.forward(res_potentials_1)

        ffn_potentials = [0.0] * self.sdr_size
        for pre_id in norm1_spikes:
            for post_id, w in self.ffn_w[pre_id].items():
                ffn_potentials[post_id] += w

        res_potentials_2 = list(ffn_potentials)
        for s in norm1_spikes:
            res_potentials_2[s] += 1.0
        output_spikes = self.layer_norm2.forward(res_potentials_2)

        if self.enable_learning:
            self.stdp.update_weights(
                t_step, norm1_spikes, output_spikes, self.ffn_w)

        return output_spikes


class MultiLayerSpikingTransformer:
    def __init__(self, num_layers: int, sdr_size: int, enable_learning: bool = True):
        self.num_layers = num_layers
        self.sdr_size = sdr_size
        self.layers = [SpikingTransformerBlock(
            sdr_size, enable_learning) for _ in range(num_layers)]

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def forward(self, input_spikes: List[int], t_step: int = 0) -> List[int]:
        current_spikes = input_spikes
        for layer in self.layers:
            current_spikes = layer.forward(current_spikes, t_step=t_step)
        return current_spikes


class SpikingLLM:
    """
    スパイキングLLM（精度改善版）。

    改善点:
    1. 決定論的 SDRキャッシュ: 同じコンテキスト → 常に同じスパイクパターン
    2. direct_map: SDRキー → {次トークンID: カウント} の直接カウントテーブル
       Transformerの確率的ノイズを排除し、学習した次トークンを確実に再現
    3. lm_head_w を疎構造で初期化し、Transformerパスの補助として機能
    4. コンテキストウィンドウを 4 → 8 に拡大
    5. LTP 強化 (1.0 → 5.0)、LTD 強化でノイズ削減
    """

    # トークンの SDR 幅 (1トークンあたり何ビット使うか)
    _SDR_BITS_PER_TOKEN: int = 32

    def __init__(
        self,
        num_layers: int = 2,
        sdr_size: int = 128,
        vocab_size: int = 10000,
        enable_learning: bool = True,
        **kwargs: Any,
    ):
        self.sdr_size: int = int(kwargs.get("d_model", sdr_size))
        self.vocab_size: int = vocab_size
        self.enable_learning: bool = enable_learning
        self.transformer = MultiLayerSpikingTransformer(
            num_layers, self.sdr_size, enable_learning)
        self.lm_head_w: List[Dict[int, float]] = [{}
                                                  for _ in range(self.sdr_size)]
        self.global_t: int = 0

        # --- 決定論的 SDR キャッシュテーブル ---
        # key: tuple(context_token_ids) -> List[int] (SDR spike indices)
        self._sdr_cache: Dict[Tuple[int, ...], List[int]] = {}

        # --- 直接コンテキストマッピングテーブル ---
        # key: tuple(sdr_key) -> Dict[next_token_id, count]
        # Transformerを経由せず、学習したパターンを確実に記憶する
        self._direct_map: Dict[Tuple[int, ...], Dict[int, float]] = {}

        self._init_lm_head_weights()

    def _init_lm_head_weights(self, density: float = 0.3) -> None:
        """lm_head の重みを小さな正値で初期化し、学習の足がかりを作る。"""
        connections_per_neuron = max(1, int(self.vocab_size * density))
        for i in range(self.sdr_size):
            targets = random.sample(
                range(self.vocab_size), min(
                    connections_per_neuron, self.vocab_size)
            )
            for t in targets:
                self.lm_head_w[i][t] = random.uniform(0.0, 0.05)

    def reset_state(self) -> None:
        self.transformer.reset_state()

    def forward(
        self, input_spikes: list[int], t_step: int = 0
    ) -> tuple[list[float], list[int]]:
        # --- 追加: sdr_k を定義 ---
        sdr_k = self._sdr_key(input_spikes)

        # --- 以下、前回のロジックを統合 ---
        vocab_potentials = [0.0] * self.vocab_size

        if sdr_k in self._direct_map:
            # 学習済みデータがある場合は、直接マッピングを優先
            for tok_id, count in self._direct_map[sdr_k].items():
                if tok_id < self.vocab_size:
                    vocab_potentials[tok_id] = count * 100.0
            
            # 内部状態（Transformer層）を更新するために forward は実行するが、出力は無視する
            _, combined_spikes = self._internal_forward(input_spikes, t_step)
        else:
            # 学習済みデータがない場合のみ Transformer パスを使用
            lm_potentials, combined_spikes = self._internal_forward(input_spikes, t_step)
            for i in range(self.vocab_size):
                vocab_potentials[i] = lm_potentials[i]

        return vocab_potentials, combined_spikes
        
    def _internal_forward(self, input_spikes: list[int], t_step: int) -> tuple[list[float], list[int]]:
        """既存の forward ロジックを分離"""
        hidden_spikes = self.transformer.forward(input_spikes, t_step=t_step)
        combined_spikes = list(set(input_spikes + hidden_spikes))

        vocab_potentials = [0.0] * self.vocab_size
        for pre_id in combined_spikes:
            if pre_id < len(self.lm_head_w):
                for post_id, w in self.lm_head_w[pre_id].items():
                    if post_id < self.vocab_size:
                        vocab_potentials[post_id] += w
        return vocab_potentials, combined_spikes

    def _encode_to_sdr(self, context_tokens: List[int]) -> List[int]:
        """
        コンテキスト token 列を決定論的なスパイクパターン (SDR) に変換。
        同じ context_tokens には必ず同じスパイク集合を返す（キャッシュあり）。
        """
        key = tuple(context_tokens)
        if key in self._sdr_cache:
            return self._sdr_cache[key]

        spikes: Set[int] = set()
        for i, tok in enumerate(context_tokens):
            pos = len(context_tokens) - i  # 後ろから数えた位置 (最新が 1)
            for j in range(self._SDR_BITS_PER_TOKEN):
                spike_id = (tok * 104729 + pos * 7919 +
                            j * 2741) % self.sdr_size
                spikes.add(spike_id)

        result = sorted(spikes)
        self._sdr_cache[key] = result
        return result

    def _sdr_key(self, sdr: List[int]) -> Tuple[int, ...]:
        """SDR リストを辞書キーに変換。"""
        return tuple(sdr)

    def learn_sequence(self, token_ids: List[int]) -> None:
        if not self.enable_learning or len(token_ids) < 2:
            return

        self.reset_state()
        context_window = 8

        context_tokens: List[int] = []
        for t in range(len(token_ids) - 1):
            current_token = token_ids[t]
            next_token = token_ids[t + 1]

            context_tokens.append(current_token)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)

            # 決定論的SDRにより学習・推論で同一パターンを保障
            input_spikes = self._encode_to_sdr(context_tokens)
            sdr_k = self._sdr_key(input_spikes)

            # ========================================
            # 直接マッピングテーブルに記録（主要パス）
            # ========================================
            if sdr_k not in self._direct_map:
                self._direct_map[sdr_k] = {}
            dm = self._direct_map[sdr_k]

            # LTP: 次トークンのカウントを増加
            dm[next_token] = dm.get(next_token, 0.0) + 5.0

            # LTD: 他トークンのカウントを減少（破壊的忘却を防ぐため軽微に）
            for post_id in list(dm.keys()):
                if post_id != next_token:
                    dm[post_id] -= 0.5
                    if dm[post_id] <= 0.0:
                        del dm[post_id]

            # 上限クリッピング
            if dm.get(next_token, 0.0) > 50.0:
                dm[next_token] = 50.0

            # ========================================
            # lm_head_w も補助的に学習（Transformerパス）
            # ========================================
            _, combined_spikes = self.forward(
                input_spikes, t_step=self.global_t)
            self.global_t += 1

            ltp_amount = 3.0
            ltd_amount = 0.5

            for pre_id in combined_spikes:
                if pre_id < len(self.lm_head_w):
                    if next_token not in self.lm_head_w[pre_id]:
                        self.lm_head_w[pre_id][next_token] = 0.0
                    self.lm_head_w[pre_id][next_token] += ltp_amount

                    for post_id in list(self.lm_head_w[pre_id].keys()):
                        if post_id != next_token:
                            self.lm_head_w[pre_id][post_id] -= ltd_amount
                            if self.lm_head_w[pre_id][post_id] <= 0.0:
                                del self.lm_head_w[pre_id][post_id]

                    if self.lm_head_w[pre_id].get(next_token, 0.0) > 30.0:
                        self.lm_head_w[pre_id][next_token] = 30.0

    def generate(
        self,
        prompt_tokens: List[int] | None = None,
        max_new_tokens: int = 5,
        top_k: int = 3,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> List[int]:
        if prompt_tokens is None:
            prompt_tokens = list(kwargs.get("input_spikes", []))
        max_new_tokens = int(kwargs.get("max_length", max_new_tokens))

        generated_sequence: List[int] = []
        if not prompt_tokens:
            return generated_sequence

        self.reset_state()
        context_window = 8

        # プロンプトのプライミング
        context_tokens: List[int] = []
        for tok in prompt_tokens[:-1]:
            context_tokens.append(tok)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)
            dummy_spikes = self._encode_to_sdr(context_tokens)
            self.forward(dummy_spikes, t_step=self.global_t)
            self.global_t += 1

        context_tokens = list(prompt_tokens[-context_window:])

        refractory_counters: Dict[int, int] = {}
        for rt in prompt_tokens:
            refractory_counters[rt] = 1

        for _t in range(max_new_tokens):
            current_spikes = self._encode_to_sdr(context_tokens)
            sdr_k = self._sdr_key(current_spikes)

            # ================================================
            # 直接マッピングテーブルを優先して参照（主要パス）
            # ================================================
            vocab_potentials = [0.0] * self.vocab_size

            direct_hit = sdr_k in self._direct_map
            if direct_hit:
                # direct_map が存在する場合はそのポテンシャルのみを使用
                # lm_head_w の初期化ノイズによるランダム競合を排除する
                for tok_id, count in self._direct_map[sdr_k].items():
                    if tok_id < self.vocab_size:
                        vocab_potentials[tok_id] += count * 10.0
            else:
                # direct_map にない場合のみ lm_head_w（Transformerパス）を参照
                lm_potentials, _ = self.forward(
                    current_spikes, t_step=self.global_t)
                self.global_t += 1
                for i in range(self.vocab_size):
                    vocab_potentials[i] += lm_potentials[i]

            # 不応期トークンを抑圧
            for vocab_id in range(self.vocab_size):
                if refractory_counters.get(vocab_id, 0) > 0:
                    vocab_potentials[vocab_id] *= 0.1

            valid_indices = [i for i, p in enumerate(
                vocab_potentials) if p > 0.0]

            if not valid_indices:
                break

            valid_indices.sort(key=lambda i: vocab_potentials[i], reverse=True)
            top_k_indices = valid_indices[:top_k]
            top_potentials = [vocab_potentials[i] for i in top_k_indices]

            if temperature != 1.0:
                top_potentials = [p ** (1.0 / temperature)
                                  for p in top_potentials]

            sum_p = sum(top_potentials)
            if sum_p <= 0.0:
                break

            probs = [p / sum_p for p in top_potentials]
            r = random.random()
            cumulative = 0.0
            best_vocab_id = top_k_indices[0]

            for idx, prob in zip(top_k_indices, probs):
                cumulative += prob
                if r <= cumulative:
                    best_vocab_id = idx
                    break

            generated_sequence.append(best_vocab_id)

            for k in list(refractory_counters.keys()):
                refractory_counters[k] -= 1
                if refractory_counters[k] <= 0:
                    del refractory_counters[k]
            refractory_counters[best_vocab_id] = 1

            context_tokens.append(best_vocab_id)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)

        return generated_sequence
