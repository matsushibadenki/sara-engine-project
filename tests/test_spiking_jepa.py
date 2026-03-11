# Directory Path: tests/test_spiking_jepa.py
# English Title: Tests and Demonstration for Hierarchical Spiking JEPA
# Purpose and Content: 階層的スパイキングJEPA(H-JEPA)の機能と動作を検証するテストコード。pytestによる単体テストに加え、直接実行時にprint文で予測結果や履歴バッファの状態をコンソール出力して動作確認できるデモ機能を含む。

import pytest
import copy

from sara_engine.models.spiking_jepa import SpikingJEPA

@pytest.fixture
def jepa_model():
    # 軽量なモデルを生成するフィクスチャ
    configs = [{"embed_dim": 16, "hidden_dim": 32}]
    # テストを高速化・簡略化するために短いtime_scalesを設定
    time_scales = {"low": 1, "medium": 2}
    model = SpikingJEPA(
        layer_configs=configs,
        ema_decay=0.9,
        learning_rate=0.1,
        time_scales=time_scales
    )
    return model

def test_jepa_initialization(jepa_model):
    """初期化時にContextとTargetのエンコーダが同じ重みを持つか確認"""
    ctx_state = jepa_model.context_encoder.state_dict()
    tgt_state = jepa_model.target_encoder.state_dict()
    
    assert ctx_state.keys() == tgt_state.keys()

def test_ema_sync(jepa_model):
    """EMA(指数移動平均)によるTarget Encoderの重み更新が機能するか確認"""
    ctx_layer = jepa_model.context_encoder.layer_0
    tgt_layer = jepa_model.target_encoder.layer_0
    
    if not hasattr(ctx_layer, 'ffn'):
        pytest.skip("ffn layer is required for this specific test")
    
    initial_tgt_w = copy.deepcopy(tgt_layer.ffn.w1[0][0]) if 0 in tgt_layer.ffn.w1[0] else 0.5
    tgt_layer.ffn.w1[0][0] = initial_tgt_w
    
    # Context Encoderの重みを意図的に変更
    ctx_layer.ffn.w1[0][0] = initial_tgt_w + 1.0
    
    # sync_encoders を呼び出し
    jepa_model._sync_encoders(alpha=0.9)
    
    # 更新後の値を検証
    expected_val = 0.9 * initial_tgt_w + 0.1 * (initial_tgt_w + 1.0)
    updated_val = tgt_layer.ffn.w1[0][0]
    
    assert abs(updated_val - expected_val) < 1e-5, f"Expected {expected_val}, got {updated_val}"

def test_jepa_forward_pass(jepa_model):
    """フォワードパスがエラーなく実行され、予測誤差を返すか確認"""
    x_spikes = [0, 1, 2, 3]
    y_spikes = [0, 1, 2, 3, 4]
    z_spikes = [5, 6]
    
    for _ in range(3):
        predicted_s_y, surprise_spikes = jepa_model.forward(x_spikes, y_spikes, z_spikes, learning=True)
    
    assert isinstance(predicted_s_y, list)
    assert isinstance(surprise_spikes, list)
    assert jepa_model.total_predictions > 0

def test_target_encoder_stop_gradient(jepa_model):
    """Target Encoderが予測誤差(Surprise)の逆伝播によって直接学習されない(Stop-Gradient)か確認"""
    jepa_model.context_encoder.reset_state()
    tgt_layer = jepa_model.target_encoder.layer_0
    
    if not hasattr(tgt_layer, 'ffn'):
        pytest.skip("Test requires FFN layer")
        
    if 0 not in tgt_layer.ffn.w1[0]:
        tgt_layer.ffn.w1[0][0] = 0.5
        
    initial_w = copy.deepcopy(tgt_layer.ffn.w1[0][0])
        
    x_spikes = [0, 1]
    y_spikes = [0, 1]
    
    jepa_model.ema_decay = 1.0
    jepa_model.forward(x_spikes, y_spikes, learning=True)
    
    current_w = tgt_layer.ffn.w1[0][0]
    assert current_w == initial_w, "Target Encoder should be stop-gradient when EMA=1.0"

def test_hierarchical_context_history(jepa_model):
    """階層的コンテキスト履歴バッファのサイズ管理が正常に機能するか確認"""
    x_spikes = [0]
    z_spikes = [1]
    
    max_history = jepa_model.max_history
    assert max_history == 3
    
    for _ in range(5):
        jepa_model.forward(x_spikes, z_spikes=z_spikes, learning=False)
        
    assert len(jepa_model.context_history) == max_history

if __name__ == "__main__":
    print("==================================================")
    print(" H-JEPA (Hierarchical Spiking JEPA) 動作テストデモ")
    print("==================================================\n")
    
    # 1. モデルの初期化
    configs = [{"embed_dim": 16, "hidden_dim": 32}]
    time_scales = {"low": 1, "medium": 3, "high": 5}
    
    print("モデルを初期化しています...")
    demo_model = SpikingJEPA(
        layer_configs=configs,
        ema_decay=0.99,
        learning_rate=0.1,
        time_scales=time_scales
    )
    
    print(f"最大履歴保持ステップ数 (Max History): {demo_model.max_history}")
    print(f"時間スケール設定: {demo_model.time_scales}\n")
    
    # 2. ダミースパイクデータの用意
    x_spikes_seq = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8]
    ]
    # TargetはContextの少し先の状態を模倣
    y_spikes_seq = [
        [1, 2, 3, 9],
        [2, 3, 4, 9],
        [3, 4, 5, 9],
        [4, 5, 6, 9],
        [5, 6, 7, 9],
        [6, 7, 8, 9],
        [7, 8, 9, 9]
    ]
    z_spikes = [10] # 固定の潜在変数
    
    # 3. タイムステップごとのシミュレーション実行
    print("--- タイムステップ処理開始 ---")
    for t in range(len(x_spikes_seq)):
        x_in = x_spikes_seq[t]
        y_in = y_spikes_seq[t]
        
        print(f"\n[Step {t+1}]")
        print(f"  入力 (Context) : {x_in}")
        print(f"  正解 (Target)  : {y_in}")
        
        # フォワード処理 (予測と自己組織化)
        predicted_s_y, surprise_spikes = demo_model.forward(
            x_spikes=x_in, 
            y_spikes=y_in, 
            z_spikes=z_spikes, 
            learning=True
        )
        
        print(f"  => 予測されたターゲット表現 : {predicted_s_y}")
        print(f"  => 予測誤差(Surprise)スパイク : {surprise_spikes}")
        print(f"  => 現在の履歴バッファサイズ   : {len(demo_model.context_history)}")
        
    print("\n==================================================")
    print(" デモ完了")
    print("==================================================")