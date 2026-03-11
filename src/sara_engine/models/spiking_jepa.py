# {
#     "//": "ディレクトリパス: src/sara_engine/models/spiking_jepa.py",
#     "//": "ファイルの日本語タイトル: 階層的スパイキングJEPA (Hierarchical Spiking JEPA)",
#     "//": "ファイルの目的や内容: SNNにおいて、H-JEPA (Hierarchical Joint-Embedding Predictive Architecture) を実装。arXiv:2602.07000に基づき、異なる時間解像度(Low, Medium, High)を持つ3レベルの予測器を用いて、長期予測安定性や細かな状態予測を実現し、スパイク空間の表現において予測符号化を用いた自己組織化を行う。"
# }

from typing import List, Dict, Optional, Tuple, Any
import copy

from ..nn.module import SNNModule
from .hierarchical_snn import HierarchicalSNN
from ..learning.predictive_coding import PredictiveCodingManager

class SpikingJEPA(SNNModule):
    """
    SNN向けの Hierarchical Joint Embedding Predictive Architecture (H-JEPA)。
    バックプロパゲーションや行列演算を使わず、予測符号化(Predictive Coding)と局所的な重み更新を利用。
    異なる時間解像度(Low, Medium, High)を持つ3つのPredictorを備え、
    過去のコンテキストから現在のターゲット埋め込みを予測する。
    Target Encoderの重みはContext EncoderのEMAとして更新される(Stop-Gradient相当)。
    """
    
    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        ema_decay: float = 0.99,
        learning_rate: float = 0.1,
        predict_threshold: float = 0.5,
        time_scales: Optional[Dict[str, int]] = None
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
        
        # 時間解像度の定義 (遅延ステップ数)
        # Low: 短期予測 (1ステップ先), Medium: 中期予測, High: 長期予測
        if time_scales is None:
            self.time_scales = {"low": 1, "medium": 3, "high": 5}
        else:
            self.time_scales = time_scales
            
        self.max_history = max(self.time_scales.values()) + 1
        
        # コンテキスト履歴バッファ: Tuple[s_xのスパイク, zのスパイク]
        self.context_history: List[Tuple[List[int], List[int]]] = []
        
        # 1. Context Encoder と Target Encoder (初期状態は同じ)
        self.context_encoder = HierarchicalSNN(copy.deepcopy(layer_configs))
        self.target_encoder = HierarchicalSNN(copy.deepcopy(layer_configs))
        
        # 内部状態を持たせるために SNNModule として登録
        self._modules["context_encoder"] = self.context_encoder
        self._modules["target_encoder"] = self.target_encoder
        
        # Target Encoderの重みをContext Encoderに同期 (初期化時)
        self._sync_encoders(alpha=0.0)
        
        # 2. 階層的 Predictor (3レベル)
        # s_x (context representation) と z (latent) または直接 s_x から s_y (target representation) へ予測
        self.predictors = {
            level: [{} for _ in range(self.embed_dim * 2)]
            for level in self.time_scales.keys()
        }
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
        H-JEPAのフォワードパス
        Args:
            x_spikes: Context 入力スパイク (現在時刻 t)
            y_spikes: Target 入力スパイク (学習時のみ使用, 現在時刻 t の正解)
            z_spikes: Latent 変数スパイク (オプショナル)
            learning: 学習を行うか
        Returns:
            predicted_s_y: Target表現のアンサンブル予測スパイク
            surprise_spikes: 予測できなかったTarget表現のスパイク (誤差)
        """
        # 1. Contextのエンコード (学習あり)
        s_x = self.context_encoder.forward(x_spikes, learning=learning)
        
        current_z = list(z_spikes) if z_spikes else []
        
        # 履歴バッファの更新
        self.context_history.append((list(s_x), current_z))
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)
            
        # 2. 階層的Predictorによる s_y (Target表現) の予測 (過去のContextから現在を予測)
        predicted_potentials: Dict[int, float] = {}
        
        for level, delay in self.time_scales.items():
            if len(self.context_history) > delay:
                # 該当する遅延の過去状態を取得
                past_s_x, past_z = self.context_history[-(delay + 1)]
            else:
                # 履歴が十分にない場合は最も古い状態を利用
                past_s_x, past_z = self.context_history[0]
                
            predictor_in = list(past_s_x)
            if past_z:
                predictor_in.extend([z + self.embed_dim for z in past_z])
                
            # 各レベルのポテンシャルをアンサンブル加算
            weights = self.predictors[level]
            for s in predictor_in:
                if s < len(weights):
                    for t, w in weights[s].items():
                        # 重みをそのまま加算し、複数の予測器からの支持を集める
                        predicted_potentials[t] = predicted_potentials.get(t, 0.0) + w

        # アンサンブル結果からの発火判定 (しきい値は予測器の数に依存せず単独でも超えられるように維持)
        predicted_s_y = [t for t, p in predicted_potentials.items() if p >= self.predict_threshold]
        
        surprise_spikes = []
        if y_spikes is not None:
            # 3. Targetのエンコード (Target Encoderは学習しない! stop-gradient)
            s_y = self.target_encoder.forward(y_spikes, learning=False)
            
            s_y_set = set(s_y)
            predicted_set = set(predicted_s_y)
            
            # アンサンブル予測の誤差 (Surprise)
            surprise_spikes = list(s_y_set - predicted_set)
            
            # メトリクス記録
            self.total_predictions += len(s_y_set)
            self.total_hits += len(s_y_set & predicted_set)
            
            if learning:
                # 4. 各レベルのPredictorの学習 (LTP / LTD)
                for level, delay in self.time_scales.items():
                    if len(self.context_history) > delay:
                        past_s_x, past_z = self.context_history[-(delay + 1)]
                    else:
                        past_s_x, past_z = self.context_history[0]
                        
                    predictor_in = list(past_s_x)
                    if past_z:
                        predictor_in.extend([z + self.embed_dim for z in past_z])
                        
                    weights = self.predictors[level]
                    
                    # 各レベル単独の予測を計算して個別に誤差を求める
                    level_potentials = {}
                    for s in predictor_in:
                        if s < len(weights):
                            for t, w in weights[s].items():
                                level_potentials[t] = level_potentials.get(t, 0.0) + w
                    
                    level_predicted_set = {t for t, p in level_potentials.items() if p >= self.predict_threshold}
                    
                    # 予測符号化による局所的な重み更新
                    PredictiveCodingManager.update_backward(
                        backward_weights=weights,
                        prev_state_spikes=predictor_in,
                        current_in_spikes=s_y,
                        predicted_in_spikes=level_predicted_set,
                        lr=self.learning_rate
                    )
                
                # 5. Target Encoder の重み更新 (EMA)
                self._sync_encoders(alpha=self.ema_decay)
                
        return predicted_s_y, surprise_spikes
        
    def reset_state(self) -> None:
        super().reset_state()
        self.context_encoder.reset_state()
        self.target_encoder.reset_state()
        self.context_history.clear()
        self.total_predictions = 0
        self.total_hits = 0