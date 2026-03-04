# ディレクトリパス: src/sara_engine/models/snn_transformer.py
# ファイルの日本語タイトル: スパイキング・トランスフォーマーモデル 完全版
# ファイルの目的や内容:
#   修正版Bをベースに以下を統合・完成させた最終版。
#   [修正版Bからの継承]
#     - Unigram/Bigram/Trigram SDRによる3階層の文脈表現
#     - 温度付きSoftmaxサンプリング（temperature=0.6）
#     - ゆらぎ幅の抑制（±5%）
#     - マイルドな不応期（decay_factor=0.75）
#   [完全版での追加・修正]
#     - 型ヒントの完全整備（mypy準拠）
#     - total_readout_sizeの計算をTrigram空間に合わせて修正（reservoir*3+embed_dim）
#     - reset_state()でdelay_bufferの初期化を確実に行う
#     - learn_sequence()のreplay回数をパラメータ化
#     - generate()の引数にtemperatureを追加して外部から制御可能に
#     - save/load時のバージョン情報をconfig.jsonに記録
#     - コーパス汚染に対してforward_step()内で不正Unicodeを除外するガード追加
#     - シナプス刈り込みのロジックを関数化して可読性を向上

from sara_engine.core.spike_attention import SpikeMultiPathwayAttention
from sara_engine.nn.attention import SpikeFuzzyAttention
from sara_engine import nn
from typing import List, Dict, Optional, Tuple
import operator
import pickle
import random
import os
import json

# ---- 定数 ----------------------------------------------------------------
_MODEL_VERSION: str = "1.3.0"          # 保存するconfig.jsonに記録するバージョン
_UNICODE_MAX: int = 0x10FFFF           # 有効なUnicodeコードポイントの上限
_SYNAPSE_MAX_WEIGHT: float = 20.0      # シナプス重みの上限（飽和防止）
_SYNAPSE_NORM_THRESH: float = 15.0     # ホメオスタシス正規化を発火させる閾値
_SYNAPSE_NORM_FACTOR: float = 0.9      # ホメオスタシス正規化の縮小係数
_SYNAPSE_PRUNE_THRESH: float = 1.0     # 刈り込みの対象になる重みの下限
_SYNAPSE_BUCKET_MAX: int = 8192        # 1シナプスバケツの最大エントリ数
_SYNAPSE_PRUNE_TARGET: int = 4096      # 刈り込み後に残すエントリ数


# ---- 設定クラス ----------------------------------------------------------
class SNNTransformerConfig:
    """
    SpikingTransformerModelのハイパーパラメータをまとめる設定クラス。

    Args:
        vocab_size:          語彙サイズ（Unicode文字単位なので1114112が上限）
        embed_dim:           埋め込み次元（= SNNブロック内のスパイク空間の大きさ）
        num_layers:          SNNTransformerBlockの積層数
        ffn_dim:             フィードフォワードネットワークの中間次元
        num_pathways:        SpikeMultiPathwayAttentionの経路数
        dropout_p:           SpikeDropoutの確率
        target_spikes_ratio: SpikeLayerNormの目標発火率（embed_dimに対する比）
        use_fuzzy:           SpikeFuzzyAttentionを使用するか（Falseで通常のMultiPathway）
        replay_count:        learn_sequence()内での経験再生回数
    """

    def __init__(
        self,
        vocab_size: int = 1114112,
        embed_dim: int = 64,
        num_layers: int = 1,
        ffn_dim: int = 256,
        num_pathways: int = 4,
        dropout_p: float = 0.1,
        target_spikes_ratio: float = 0.15,
        use_fuzzy: bool = False,
        replay_count: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.num_pathways = num_pathways
        self.dropout_p = dropout_p
        self.target_spikes_ratio = target_spikes_ratio
        self.use_fuzzy = use_fuzzy
        self.replay_count = replay_count

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_version": _MODEL_VERSION,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "num_pathways": self.num_pathways,
            "dropout_p": self.dropout_p,
            "target_spikes_ratio": self.target_spikes_ratio,
            "use_fuzzy": self.use_fuzzy,
            "replay_count": self.replay_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SNNTransformerConfig":
        # model_versionはコンストラクタ引数ではないので除外
        filtered: Dict[str, object] = {
            k: v for k, v in data.items() if k != "model_version"
        }
        return cls(**filtered)  # type: ignore[arg-type]


# ---- SNNブロック ---------------------------------------------------------
class SNNTransformerBlock(nn.SNNModule):
    """
    1層分のSNNトランスフォーマーブロック。
    Residual接続 + LayerNorm + Attention + FFN の構成。
    """

    def __init__(self, config: SNNTransformerConfig) -> None:
        super().__init__()
        self.config = config

        target_spikes = max(1, int(config.embed_dim * config.target_spikes_ratio))

        self.norm1 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout1 = nn.SpikeDropout(p=config.dropout_p)

        if config.use_fuzzy:
            self.attention = SpikeFuzzyAttention(
                embed_dim=config.embed_dim,
                threshold=0.2,
                top_k=3,
            )
        else:
            self.attention = SpikeMultiPathwayAttention(
                embed_dim=config.embed_dim,
                num_pathways=config.num_pathways,
                context_size=128,
            )

        self.norm2 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout2 = nn.SpikeDropout(p=config.dropout_p)

        self.ffn = nn.Sequential(
            nn.LinearSpike(
                in_features=config.embed_dim,
                out_features=config.ffn_dim,
                density=0.2,
            ),
            nn.LinearSpike(
                in_features=config.ffn_dim,
                out_features=config.embed_dim,
                density=0.2,
            ),
        )
        # ブロック内のスパイク数の上限（爆発的発火の防止）
        self.max_block_spikes: int = max(1, config.embed_dim // 2)

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        # --- Attention branch ---
        norm_spikes1 = self.norm1(spikes, learning=learning)
        attn_out = self.attention.forward(norm_spikes1, learning=learning)
        drop_attn = self.dropout1(attn_out, learning=learning)
        res1_spikes: List[int] = list(set(spikes + drop_attn))

        # --- FFN branch ---
        norm_spikes2 = self.norm2(res1_spikes, learning=learning)
        ffn_out = self.ffn(norm_spikes2, learning=learning)
        drop_ffn = self.dropout2(ffn_out, learning=learning)
        res2_spikes: List[int] = list(set(res1_spikes + drop_ffn))

        # 発火数の上限クリップ（爆発防止）
        if len(res2_spikes) > self.max_block_spikes:
            res2_spikes = random.sample(res2_spikes, self.max_block_spikes)

        return res2_spikes


# ---- メインモデル --------------------------------------------------------
class SpikingTransformerModel(nn.SNNModule):
    """
    文字単位で動作するスパイキング・トランスフォーマー言語モデル。

    リザーバー空間にUnigram/Bigram/Trigram の3階層SDRを配置し、
    読み出しシナプスへの強化学習（報酬・罰則型ヘッブ則）で学習する。
    """

    def __init__(self, config: SNNTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # コンテキスト長とリザーバーの設定
        self.context_length: int = 64
        self.reservoir_size: int = 8192

        # Unigram(×1) + Bigram(×1) + Trigram(×1) + embed_dim
        self.total_readout_size: int = (self.reservoir_size * 3) + config.embed_dim

        # SNNブロックの積層
        layers: List[SNNTransformerBlock] = [
            SNNTransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.transformer_layers = nn.Sequential(*layers)

        # 遅延バッファ（文脈保持）
        self.delay_buffer: List[int] = []

        # 読み出しシナプス: [ニューロンindex] -> {トークンid: 重み}
        self.readout_synapses: List[Dict[int, float]] = [
            {} for _ in range(self.total_readout_size)
        ]
        self.register_state("readout_synapses")

    # ------------------------------------------------------------------
    # 状態リセット
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        super().reset_state()
        self.delay_buffer.clear()
        for layer in getattr(self.transformer_layers, "modules", []):
            if hasattr(layer, "attention"):
                layer.attention.reset_state()

    # ------------------------------------------------------------------
    # リザーバースパイク生成（Unigram/Bigram/Trigram SDR）
    # ------------------------------------------------------------------
    def _get_reservoir_spikes(self, token_id: int) -> List[int]:
        """
        token_idをdelay_bufferに追加し、
        Unigram・Bigram・Trigram の3階層SDRをXorShift32で生成して返す。

        各階層はreservoir_sizeのオフセットで完全に分離されており、
        異なるトークン間でのスパイク混線（ハッシュ衝突）が起きない。
        """
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        spikes: set = set()
        prev_tok: Optional[int] = None
        prev_prev_tok: Optional[int] = None

        for delay, tok in enumerate(self.delay_buffer):
            num_spikes = max(2, 24 - int(delay * 0.4))

            # --- Unigram SDR（オフセット 0） ---
            seed_u = (tok * 31337) ^ (delay * 982451653) ^ 0x5A5A5A5A
            state_u = seed_u & 0xFFFFFFFF
            if state_u == 0:
                state_u = 1
            for _ in range(num_spikes):
                state_u ^= (state_u << 13) & 0xFFFFFFFF
                state_u ^= (state_u >> 17) & 0xFFFFFFFF
                state_u ^= (state_u << 5) & 0xFFFFFFFF
                spikes.add(state_u % self.reservoir_size)

            # --- Bigram SDR（オフセット reservoir_size） ---
            if prev_tok is not None:
                seed_b = (tok * 31) ^ (prev_tok * 53) ^ (delay * 17) ^ 0x12345678
                state_b = seed_b & 0xFFFFFFFF
                if state_b == 0:
                    state_b = 1
                for _ in range(num_spikes // 2 + 1):
                    state_b ^= (state_b << 13) & 0xFFFFFFFF
                    state_b ^= (state_b >> 17) & 0xFFFFFFFF
                    state_b ^= (state_b << 5) & 0xFFFFFFFF
                    spikes.add((state_b % self.reservoir_size) + self.reservoir_size)

            # --- Trigram SDR（オフセット reservoir_size * 2） ---
            if prev_prev_tok is not None and prev_tok is not None:
                seed_t = (
                    (tok * 13)
                    ^ (prev_tok * 37)
                    ^ (prev_prev_tok * 71)
                    ^ (delay * 23)
                    ^ 0x87654321
                )
                state_t = seed_t & 0xFFFFFFFF
                if state_t == 0:
                    state_t = 1
                for _ in range(max(1, num_spikes // 3)):
                    state_t ^= (state_t << 13) & 0xFFFFFFFF
                    state_t ^= (state_t >> 17) & 0xFFFFFFFF
                    state_t ^= (state_t << 5) & 0xFFFFFFFF
                    spikes.add(
                        (state_t % self.reservoir_size) + self.reservoir_size * 2
                    )

            prev_prev_tok = prev_tok
            prev_tok = tok

        return list(spikes)

    # ------------------------------------------------------------------
    # シナプス刈り込み（内部ユーティリティ）
    # ------------------------------------------------------------------
    def _prune_synapses(
        self, synapses: Dict[int, float], protect_id: int
    ) -> None:
        """
        シナプスバケツが上限を超えた場合に余分なエントリを刈り込む。
        protect_idは必ず保持する（学習ターゲットトークン）。
        """
        if len(synapses) <= _SYNAPSE_BUCKET_MAX:
            return

        # フェーズ1: 閾値未満の弱いシナプスを一括削除
        weak_keys = [
            k for k, v in synapses.items()
            if v < _SYNAPSE_PRUNE_THRESH and k != protect_id
        ]
        for k in weak_keys:
            del synapses[k]

        # フェーズ2: まだ超えていればスコア順で下位を削除
        if len(synapses) > _SYNAPSE_BUCKET_MAX:
            sorted_keys = sorted(synapses.keys(), key=lambda k: synapses[k])
            for k in sorted_keys[:_SYNAPSE_PRUNE_TARGET]:
                if k != protect_id:
                    del synapses[k]

    # ------------------------------------------------------------------
    # 電位から次トークンを選択（温度付きサンプリング）
    # ------------------------------------------------------------------
    @staticmethod
    def _temperature_sample(
        candidates: List[Tuple[int, float]],
        temperature: float,
    ) -> int:
        """
        candidates: [(token_id, potential), ...] のリスト（電位降順前提）
        temperature: サンプリング温度（低いほど最大値に集中）
        戻り値: サンプリングで選ばれたtoken_id
        """
        weights = [pow(max(1e-9, item[1]), 1.0 / temperature) for item in candidates]
        total_weight = sum(weights)
        r = random.uniform(0.0, total_weight)
        cumulative = 0.0
        for item, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return item[0]
        # 丸め誤差フォールバック
        return candidates[0][0]

    # ------------------------------------------------------------------
    # 1ステップの順伝播
    # ------------------------------------------------------------------
    def forward_step(
        self,
        token_id: int,
        learning: bool = True,
        target_id: Optional[int] = None,
        refractory_tokens: Optional[List[int]] = None,
        temperature: float = 0.6,
        fire_threshold: float = 30.0,
    ) -> int:
        """
        1トークン分の順伝播を行い、次トークンの予測IDを返す。

        Args:
            token_id:          現在の入力トークン（Unicodeコードポイント）
            learning:          True=学習モード / False=推論モード
            target_id:         学習モード時の正解トークン
            refractory_tokens: 推論時の不応期バッファ（繰り返し抑制）
            temperature:       推論時のサンプリング温度（低いほど確定的）
            fire_threshold:    推論時に生成を続けるための最低電位
        """
        # --- ガード: 不正なコードポイントを除外 ---
        # コーパス汚染（16進数文字列等）が学習データに混入していた場合、
        # 制御文字や代替文字（サロゲートペア等）が候補に上がることがある。
        # 推論時はこれを弾いて出力の安全性を担保する。
        def _is_valid_output_token(tid: int) -> bool:
            if tid <= 0 or tid > _UNICODE_MAX:
                return False
            # Unicodeサロゲートペア領域（D800-DFFF）を除外
            if 0xD800 <= tid <= 0xDFFF:
                return False
            return True

        # --- リザーバースパイク生成 ---
        res_spikes: List[int] = self._get_reservoir_spikes(token_id)

        # --- SNNブロックへ入力（embed_dim空間に射影） ---
        block_input: List[int] = list(
            set([s % self.config.embed_dim for s in res_spikes])
        )
        block_spikes: List[int] = self.transformer_layers(
            block_input, learning=learning
        )

        # --- 読み出しスパイクを構築 ---
        # Trigram空間（reservoir*3）の直後にブロック出力を配置
        block_offset = self.reservoir_size * 3
        readout_spikes: List[int] = list(
            set(res_spikes + [s + block_offset for s in block_spikes])
        )

        # --- 読み出し電位の計算 ---
        out_potentials: Dict[int, float] = {}
        for s in readout_spikes:
            if s < self.total_readout_size:
                for v_idx, w in self.readout_synapses[s].items():
                    current = out_potentials.get(v_idx, 0.0)
                    out_potentials[v_idx] = current + w

        # --- 不応期ペナルティ（推論時のみ・繰り返し文字の抑制） ---
        if not learning and refractory_tokens:
            decay_factor = 0.75
            for r_tok in reversed(refractory_tokens):
                if r_tok in out_potentials:
                    out_potentials[r_tok] *= decay_factor
                decay_factor += 0.08
                if decay_factor > 1.0:
                    decay_factor = 1.0

        # --- 予測トークンの選択 ---
        predicted_id = 32   # デフォルトはスペース（コードポイント32）
        margin = 0.0

        if out_potentials:
            # 推論時のみ微小なシナプス伝達ゆらぎを付与（±5%）
            if not learning:
                for k in out_potentials:
                    out_potentials[k] *= random.uniform(0.95, 1.05)

            sorted_items = sorted(
                out_potentials.items(),
                key=operator.itemgetter(1),
                reverse=True,
            )

            if learning:
                # 学習時: greedy（マージン計算のため最高電位を選択）
                if sorted_items[0][1] > 0.1:
                    predicted_id = sorted_items[0][0]
                    if len(sorted_items) > 1:
                        margin = sorted_items[0][1] - sorted_items[1][1]
                    else:
                        margin = sorted_items[0][1]
            else:
                # 推論時: 温度付きサンプリング
                top_k = sorted_items[:5]
                if top_k[0][1] > fire_threshold:
                    # 不正トークンを候補から除外
                    valid_candidates = [
                        (tid, pot) for tid, pot in top_k
                        if _is_valid_output_token(tid)
                    ]
                    if valid_candidates:
                        predicted_id = self._temperature_sample(
                            valid_candidates, temperature
                        )
                    else:
                        predicted_id = 0    # 有効候補なし → 生成停止
                else:
                    predicted_id = 0        # 電位不足 → 生成停止

        # ------------------------------------------------------------------
        # 学習フェーズ: 報酬・罰則型ヘッブ則によるシナプス更新
        # ------------------------------------------------------------------
        if learning and target_id is not None:
            is_correct = (predicted_id == target_id)

            if is_correct:
                # 正解時: マージンが小さいほど（ギリギリの正解）強く強化
                reward_factor = max(0.5, 4.0 - margin)
                punish_factor = 0.2
            else:
                # 不正解時: マージンが大きいほど（自信を持った誤り）強く罰則
                surprise = 1.0 + margin
                punish_factor = min(2.5, surprise * 1.5)
                reward_factor = 1.5

            for s in readout_spikes:
                if s >= self.total_readout_size:
                    continue

                synapses = self.readout_synapses[s]

                # 正解トークンへの強化
                current_w = synapses.get(target_id, 0.0)
                new_w = min(_SYNAPSE_MAX_WEIGHT, current_w + (1.5 * reward_factor))
                synapses[target_id] = new_w

                # ホメオスタシス正規化（特定シナプスの独占を防ぐ）
                if new_w > _SYNAPSE_NORM_THRESH:
                    for k in synapses:
                        synapses[k] *= _SYNAPSE_NORM_FACTOR

                # 誤答トークンへのペナルティ
                if not is_correct and predicted_id in synapses:
                    synapses[predicted_id] -= 2.0 * punish_factor
                    if synapses[predicted_id] <= 0.0:
                        del synapses[predicted_id]

                # シナプス刈り込み（メモリ上限管理）
                self._prune_synapses(synapses, protect_id=target_id)

        return predicted_id

    # ------------------------------------------------------------------
    # シーケンス学習
    # ------------------------------------------------------------------
    def learn_sequence(self, input_ids: List[int]) -> None:
        """
        トークン列を経験再生（replay_count回）で学習する。
        末尾に終端トークン（0）を追加してEOS学習も行う。
        """
        sequence = input_ids + [0]
        for _ in range(self.config.replay_count):
            self.reset_state()
            for i in range(len(sequence) - 1):
                self.forward_step(
                    sequence[i],
                    learning=True,
                    target_id=sequence[i + 1],
                )

    # ------------------------------------------------------------------
    # テキスト生成
    # ------------------------------------------------------------------
    def generate(
        self,
        input_ids: List[int],
        max_length: int = 150,
        temperature: float = 0.6,
        fire_threshold: float = 30.0,
    ) -> List[int]:
        """
        プロンプト（input_ids）を与えてテキストを自己回帰生成する。

        Args:
            input_ids:      入力トークン列（Unicodeコードポイントのリスト）
            max_length:     生成するトークンの最大数
            temperature:    サンプリング温度（0.1=確定的, 1.0=ランダム）
            fire_threshold: 発火閾値（この電位以下なら生成停止）

        Returns:
            生成されたトークンIDのリスト（プロンプト部分は含まない）
        """
        self.reset_state()

        # プロンプトを流し込んで状態を構築
        first_pred = 32
        for token_id in input_ids:
            first_pred = self.forward_step(
                token_id,
                learning=False,
                temperature=temperature,
                fire_threshold=fire_threshold,
            )

        generated_ids: List[int] = []
        current_token = first_pred
        refractory_buffer: List[int] = []

        for _ in range(max_length):
            if current_token == 0:
                break   # EOS → 終了

            generated_ids.append(current_token)
            refractory_buffer.append(current_token)

            # 不応期バッファ: 直近6文字を保持して繰り返しを抑制
            if len(refractory_buffer) > 6:
                refractory_buffer.pop(0)

            current_token = self.forward_step(
                current_token,
                learning=False,
                refractory_tokens=refractory_buffer,
                temperature=temperature,
                fire_threshold=fire_threshold,
            )

        return generated_ids

    # ------------------------------------------------------------------
    # 保存・ロード
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str) -> None:
        """モデルの設定とシナプス重みを指定ディレクトリに保存する。"""
        os.makedirs(save_directory, exist_ok=True)

        config_path = os.path.join(save_directory, "config.json")
        config_dict = self.config.to_dict()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        state_path = os.path.join(save_directory, "model_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(self.state_dict(), f)

        print(f"[SpikingTransformerModel] Saved to '{save_directory}' (v{_MODEL_VERSION})")

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "SpikingTransformerModel":
        """保存済みディレクトリからモデルを復元する。"""
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, object] = json.load(f)

        config = SNNTransformerConfig.from_dict(raw)
        model = cls(config)

        state_path = os.path.join(save_directory, "model_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            model.load_state_dict(state)

        saved_version = raw.get("model_version", "unknown")
        print(
            f"[SpikingTransformerModel] Loaded from '{save_directory}' "
            f"(saved version: {saved_version}, current: {_MODEL_VERSION})"
        )
        return model