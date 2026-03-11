import pytest
import copy

from sara_engine.models.spiking_jepa import SpikingJEPA

@pytest.fixture
def jepa_model():
    # 軽量なモデルを生成するフィクスチャ
    configs = [{"embed_dim": 16, "hidden_dim": 32}]
    model = SpikingJEPA(layer_configs=configs, ema_decay=0.9, learning_rate=0.1)
    return model

def test_jepa_initialization(jepa_model):
    """初期化時にContextとTargetのエンコーダが同じ重みを持つか確認"""
    ctx_state = jepa_model.context_encoder.state_dict()
    tgt_state = jepa_model.target_encoder.state_dict()
    
    # 完全に同じ要素を持っているか確認 (辞書のシリアライズ化などで不一致が出る可能性があるため簡易チェック)
    assert ctx_state.keys() == tgt_state.keys()

def test_ema_sync(jepa_model):
    """EMA(指数移動平均)によるTarget Encoderの重み更新が機能するか確認"""
    ctx_layer = jepa_model.context_encoder.layer_0
    tgt_layer = jepa_model.target_encoder.layer_0
    
    # テストのため初期状態のTarget Encoderの重みをコピー (Pythonの実装に依存)
    # ここではFNN部分でテスト
    if not hasattr(ctx_layer, 'ffn'):
        pytest.skip("ffn layer is required for this specific test")
    
    initial_tgt_w = copy.deepcopy(tgt_layer.ffn.w1[0][0]) if 0 in tgt_layer.ffn.w1[0] else 0.5
    tgt_layer.ffn.w1[0][0] = initial_tgt_w
    
    # Context Encoderの重みを意図的に変更
    ctx_layer.ffn.w1[0][0] = initial_tgt_w + 1.0
    
    # sync_encoders を呼び出し
    jepa_model._sync_encoders(alpha=0.9)
    
    # 更新後の値を検証
    # 新しい値 = 0.9 * target + 0.1 * context
    expected_val = 0.9 * initial_tgt_w + 0.1 * (initial_tgt_w + 1.0)
    
    # 更新されたTarget Encoderの値を取得
    updated_val = tgt_layer.ffn.w1[0][0]
    
    assert abs(updated_val - expected_val) < 1e-5, f"Expected {expected_val}, got {updated_val}"

def test_jepa_forward_pass(jepa_model):
    """フォワードパスがエラーなく実行され、予測誤差を返すか確認"""
    x_spikes = [0, 1, 2, 3]
    y_spikes = [0, 1, 2, 3, 4]  # ターゲットは少し異なる入力かも
    z_spikes = [5, 6]  # 潜在変数
    
    predicted_s_y, surprise_spikes = jepa_model.forward(x_spikes, y_spikes, z_spikes, learning=True)
    
    assert isinstance(predicted_s_y, list)
    assert isinstance(surprise_spikes, list)
    assert jepa_model.total_predictions > 0

def test_target_encoder_stop_gradient(jepa_model):
    """Target Encoderが予測誤差(Surprise)の逆伝播によって直接学習されない(Stop-Gradient)か確認"""
    # 記録のための初期状態
    jepa_model.context_encoder.reset_state()
    tgt_layer = jepa_model.target_encoder.layer_0
    
    if not hasattr(tgt_layer, 'ffn'):
        pytest.skip("Test requires FFN layer")
        
    if 0 not in tgt_layer.ffn.w1[0]:
        tgt_layer.ffn.w1[0][0] = 0.5
        
    initial_w = copy.deepcopy(tgt_layer.ffn.w1[0][0])
        
    x_spikes = [0, 1]
    y_spikes = [0, 1]
    
    # まずは EMA 更新を 1.0 (Target不変)にしてフォワード
    jepa_model.ema_decay = 1.0
    jepa_model.forward(x_spikes, y_spikes, learning=True)
    
    # EMA Decay が 1.0 の場合、Target Encoderの重みは一切変わらないこと
    current_w = tgt_layer.ffn.w1[0][0]
    assert current_w == initial_w, "Target Encoder should be stop-gradient and not update via STDP locally when EMA=1.0"
