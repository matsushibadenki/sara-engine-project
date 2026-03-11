# {
#     "//": "ディレクトリパス: src/sara_engine/models/spiking_jepa.py",
#     "//": "ファイルの日本語タイトル: スパイキングJEPA (Joint Embedding Predictive Architecture)",
#     "//": "ファイルの目的や内容: SNNにおいて、生データ空間ではなく「埋め込み(スパイク表現)空間」での予測を行うJEPAアーキテクチャ. Context EncoderとTarget Encoderを備え、Predictorを用いて表現予測と予測誤差(Surprise)に基づく自己組織化を実現する。"
# }

from typing import List, Dict, Optional, Tuple, Any
import copy

from ..nn.module import SNNModule
from .hierarchical_snn import HierarchicalSNN
from ..learning.predictive_coding import PredictiveCodingManager

class SpikingJEPA(SNNModule):
    """
    SNN向けの Joint Embedding Predictive Architecture (JEPA)。
    バックプロパゲーションではなく予測符号化(Predictive Coding)に基づく局所学習を利用。
    Target Encoderの重みはContext EncoderのEMAとして更新される(Stop-Gradient相当)。
    """
    
    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        ema_decay: float = 0.99,
        learning_rate: float = 0.1,
        predict_threshold: float = 0.5
    ):
        super().__init__()
        self.layer_configs = layer_configs
        if not layer_configs:
            raise ValueError("layer_configs must not be empty.")
            
        # 最終層の埋め込み次元を予測器のために保持
        self.embed_dim = layer_configs[-1]["embed_dim"]
        self.ema_decay = ema_decay
        self.learning_rate = learning_rate
        self.predict_threshold = predict_threshold
        
        # 1. Context Encoder と Target Encoder (初期状態は同じ)
        self.context_encoder = HierarchicalSNN(copy.deepcopy(layer_configs))
        self.target_encoder = HierarchicalSNN(copy.deepcopy(layer_configs))
        
        # 内部状態を持たせるために SNNModule として登録
        self._modules["context_encoder"] = self.context_encoder
        self._modules["target_encoder"] = self.target_encoder
        
        # Target Encoderの重みをContext Encoderに同期 (初期化時)
        self._sync_encoders(alpha=0.0)
        
        # 2. Predictor (Predictive Coding Manager を利用)
        # s_x (context representation) と z (latent) または直接 s_x から s_y (target representation) へ予測
        self.predictor_weights: List[Dict[int, float]] = [{} for _ in range(self.embed_dim * 2)]
        self.predictive_manager = PredictiveCodingManager(learning_rate=learning_rate)
        
        # 潜在変数 z 用の次元 (今回はランダムスパイクとして注入可能とする)
        self.latent_dim = self.embed_dim

        # メトリクス
        self.total_predictions = 0
        self.total_hits = 0

    def _blend_state_dicts(self, dict_tgt: Dict[str, Any], dict_ctx: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        """再帰的にTarget(tgt)とContext(ctx)の状態辞書をEMAブレンドする"""
        result = {}
        all_keys = set(dict_tgt.keys()) | set(dict_ctx.keys())
        
        for k in all_keys:
            if k not in dict_ctx:
                result[k] = copy.deepcopy(dict_tgt[k])
                continue
            if k not in dict_tgt:
                result[k] = copy.deepcopy(dict_ctx[k])
                continue
                
            v_ctx = dict_ctx[k]
            v_tgt = dict_tgt[k]
            
            if isinstance(v_ctx, dict) and isinstance(v_tgt, dict):
                # 再帰的に辞書をブレンド
                result[k] = self._blend_state_dicts(v_tgt, v_ctx, alpha)
            elif isinstance(v_ctx, list) and isinstance(v_tgt, list):
                # 全要素ごとにブレンド（中身が辞書や数値の場合）
                blended_list = []
                # リストサイズは大きい方に合わせる
                max_len = max(len(v_ctx), len(v_tgt))
                for idx in range(max_len):
                    if idx < len(v_ctx) and idx < len(v_tgt):
                        item_ctx = v_ctx[idx]
                        item_tgt = v_tgt[idx]
                        if isinstance(item_ctx, dict) and isinstance(item_tgt, dict):
                            blended_list.append(self._blend_state_dicts(item_tgt, item_ctx, alpha))
                        elif isinstance(item_ctx, (float, int)) and isinstance(item_tgt, (float, int)):
                            if isinstance(item_ctx, bool):  # boolは数値判定されるため弾く
                                blended_list.append(item_tgt)
                            else:
                                blended_list.append(alpha * item_tgt + (1.0 - alpha) * item_ctx)
                        else:
                            blended_list.append(copy.deepcopy(item_tgt))
                    elif idx < len(v_ctx):
                        blended_list.append(copy.deepcopy(v_ctx[idx]))
                    else:
                        blended_list.append(copy.deepcopy(v_tgt[idx]))
                result[k] = blended_list
            elif isinstance(v_ctx, (float, int)) and isinstance(v_tgt, (float, int)):
                if isinstance(v_ctx, bool):
                    result[k] = v_tgt
                else:
                    result[k] = alpha * v_tgt + (1.0 - alpha) * v_ctx
            else:
                result[k] = copy.deepcopy(v_tgt)
        return result

    def _sync_encoders(self, alpha: float) -> None:
        """Target Encoderの重みをContext EncoderのEMAで更新する。
        alpha = 0.0 なら完全上書き、alpha = 0.99 ならゆっくり更新。
        """
        for i in range(len(self.context_encoder.layer_configs)):
            ctx_layer = getattr(self.context_encoder, f"layer_{i}")
            tgt_layer = getattr(self.target_encoder, f"layer_{i}")
            
            if hasattr(ctx_layer, "state_dict") and hasattr(tgt_layer, "state_dict"):
                ctx_state = ctx_layer.state_dict()
                tgt_state = tgt_layer.state_dict()
                
                blended_state = self._blend_state_dicts(tgt_state, ctx_state, alpha)
                if hasattr(tgt_layer, "load_state_dict"):
                    tgt_layer.load_state_dict(blended_state)

    def forward(self, x_spikes: List[int], y_spikes: Optional[List[int]] = None, z_spikes: Optional[List[int]] = None, learning: bool = True) -> Tuple[List[int], List[int]]:
        """
        JEPAのフォワードパス
        Args:
            x_spikes: Context 入力スパイク
            y_spikes: Target 入力スパイク (学習時のみ使用)
            z_spikes: Latent 変数スパイク (オプショナル)
            learning: 学習を行うか
        Returns:
            predicted_s_y: Target表現の予測スパイク
            surprise_spikes: 予測できなかったTarget表現のスパイク (誤差)
        """
        # 1. Contextのエンコード (学習あり)
        s_x = self.context_encoder.forward(x_spikes, learning=learning)
        
        # Predictorへの入力 (s_x と z_spikes の結合)
        predictor_in = list(s_x)
        if z_spikes:
            # 潜在変数をオフセットして結合 (インデックスが被らないように)
            predictor_in.extend([z + self.embed_dim for z in z_spikes])
            
        # 2. s_y (Target表現)の予測
        potentials: Dict[int, float] = {}
        for s in predictor_in:
            if s < len(self.predictor_weights):
                for t, w in self.predictor_weights[s].items():
                    potentials[t] = potentials.get(t, 0.0) + w
                    
        predicted_s_y = [t for t, p in potentials.items() if p >= self.predict_threshold]
        
        surprise_spikes = []
        if y_spikes is not None:
            # 3. Targetのエンコード (Target Encoderは学習しない! stop-gradient)
            s_y = self.target_encoder.forward(y_spikes, learning=False)
            
            s_y_set = set(s_y)
            predicted_set = set(predicted_s_y)
            
            # Predictorの誤差 (Surprise)
            surprise_spikes = list(s_y_set - predicted_set)
            
            # メトリクス記録
            self.total_predictions += len(s_y_set)
            self.total_hits += len(s_y_set & predicted_set)
            
            if learning:
                # 4. Predictor の学習 (LTP / LTD) - s_y を目標(current_in_spikes)としてPredictiveCodingの逆モデルと同様に更新
                PredictiveCodingManager.update_backward(
                    backward_weights=self.predictor_weights,
                    prev_state_spikes=predictor_in,
                    current_in_spikes=s_y,
                    predicted_in_spikes=predicted_set,
                    lr=self.learning_rate
                )
                
                # Context Encoder側への勾配流(予測誤差による更新)に相当する局所学習を追加する場合:
                # Surpriseを Context Encoderへの Target Propagation 信号として使うことも可能だが、
                # ここではまず Context Encoder自体のBottom-Up自己組織化(上述forward(learning=True))とEMAに依存する標準JEPAを模倣。
                
                # 5. Target Encoder の重み更新 (EMA)
                self._sync_encoders(alpha=self.ema_decay)
                
        return predicted_s_y, surprise_spikes
        
    def reset_state(self) -> None:
        super().reset_state()
        self.context_encoder.reset_state()
        self.target_encoder.reset_state()
        self.total_predictions = 0
        self.total_hits = 0
