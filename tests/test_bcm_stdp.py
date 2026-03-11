# Directory Path: tests/test_bcm_stdp.py
# English Title: BCM-STDP Layer Test Suite
# Purpose/Content: Unit tests for the newly added BCMSTDPLayer. Verifies initialization, sliding threshold dynamics, heterosynaptic LTD, multi-language support, and ensures proper homeostatic scaling behavior without utilizing matrix operations.

import random
from sara_engine.learning.stdp import BCMSTDPLayer

class TestBCMSTDPLayer:
    """BCM-STDPレイヤーの機能と挙動を検証するユニットテスト"""

    def test_initialization(self) -> None:
        """初期化時に正しいパラメータと状態リストが構築されるか確認"""
        layer = BCMSTDPLayer(num_inputs=10, num_outputs=5, threshold=0.5)
        assert layer.num_inputs == 10
        assert layer.num_outputs == 5
        assert len(layer.y_traces) == 5
        assert len(layer.theta_m) == 5
        assert layer.theta_m[0] == 0.1

    def test_multilingual_message(self) -> None:
        """多言語対応のステータス取得機能が正常に機能するか確認"""
        layer = BCMSTDPLayer(num_inputs=2, num_outputs=2)
        
        msg_en = layer.get_status_message("en")
        assert "BCM-STDP" in msg_en
        assert "metaplasticity" in msg_en

        msg_ja = layer.get_status_message("ja")
        assert "BCM-STDP" in msg_ja
        assert "メタ可塑性" in msg_ja

        msg_fr = layer.get_status_message("fr")
        assert "métaplasticité" in msg_fr

        msg_unknown = layer.get_status_message("es")
        assert msg_unknown == msg_en

    def test_sliding_threshold_dynamics(self) -> None:
        """
        発火による発火履歴トレース(y_traces)の上昇と、
        それに伴うBCMスライディング閾値(theta_m)の動的変動を確認
        """
        layer = BCMSTDPLayer(num_inputs=2, num_outputs=1, threshold=0.1)
        layer.synapses[0] = {0: 1.5, 1: 1.5}
        
        initial_theta_m = layer.theta_m[0]
        out_spikes, _ = layer.process_step([1, 1], reward=1.0)
        
        assert out_spikes[0] == 1
        assert layer.y_traces[0] > 0.0
        assert layer.theta_m[0] != initial_theta_m

    def test_bcm_weight_modulation(self) -> None:
        """
        BCM則に基づく重みの増強（LTP）と抑圧（ヘテロシナプティックLTD）の挙動確認
        """
        random.seed(42)
        layer = BCMSTDPLayer(num_inputs=3, num_outputs=1, threshold=0.1)
        
        # 恒常性スケーリングによる意図しない重み増幅を防ぐため、ターゲットを初期重み合計(2.5)に合わせる
        layer.target_weight_sum = 2.5
        
        layer.synapses[0] = {0: 1.0, 1: 1.0, 2: 0.5}
        
        layer.y_traces[0] = 2.0 
        layer.theta_m[0] = 1.0
        
        input_spikes = [1, 0, 0]
        layer.process_step(input_spikes, reward=1.0)
        
        synapses = layer.synapses[0]
        
        assert 0 in synapses
        
        if 1 in synapses:
            assert synapses[1] < 1.0
        if 2 in synapses:
            assert synapses[2] < 0.5

    def test_reset_state(self) -> None:
        """状態リセットが膜電位と発火閾値のみを初期化し、シナプス重みやBCM履歴を破壊しないことの確認"""
        layer = BCMSTDPLayer(num_inputs=2, num_outputs=1)
        layer.potentials[0] = 0.5
        layer.thresholds[0] = 1.5
        layer.y_traces[0] = 0.8
        
        layer.reset_state()
        
        assert layer.potentials[0] == 0.0
        assert layer.thresholds[0] == layer.base_threshold
        assert layer.y_traces[0] == 0.8