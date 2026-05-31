# Directory Path: tests/test_spiking_jepa.py
# English Title: Spiking JEPA Test Suite
# Purpose/Content: Unit tests for the backpropagation-free Spiking JEPA module. Verifies multi-language support, weight sparsity, and the ability to learn predictive representations (energy minimization) using pure list operations.

import random
from sara_engine.models.spiking_jepa import SpikingJEPA, EnergyMinimizer

class TestEnergyMinimizer:
    """誤差逆伝播の代替となる局所エネルギー（Surprise）計算のテスト"""

    def test_compute_surprise_signal_perfect_match(self) -> None:
        """予測と目標が完全に一致した場合、最大のポジティブシグナルが返る"""
        minimizer = EnergyMinimizer(size=5)
        pred = [1, 0, 1, 0, 0]
        targ = [1, 0, 1, 0, 0]
        
        surprise, signal = minimizer.compute_surprise_signal(pred, targ)
        
        # 予測が一致しているのでSurprise(驚き)スパイクは出ないはず
        assert sum(surprise) == 0
        # 精度100%なのでシグナルは +1.0
        assert signal == 1.0

    def test_compute_surprise_signal_complete_mismatch(self) -> None:
        """予測が完全に外れた場合、ネガティブシグナルとSurpriseスパイクが返る"""
        minimizer = EnergyMinimizer(size=5)
        # 目標は発火しているが、予測は全く発火していない状態
        pred = [0, 0, 0, 0, 0]
        targ = [1, 1, 1, 0, 0]
        
        surprise, signal = minimizer.compute_surprise_signal(pred, targ)
        
        # 目標が発火した3箇所でSurpriseスパイクが出る
        assert sum(surprise) == 3
        # 精度0%なのでシグナルは -1.0
        assert signal == -1.0

    def test_compute_surprise_signal_no_target(self) -> None:
        """目標が全く発火していない(無音)状態では学習シグナルは0になる"""
        minimizer = EnergyMinimizer(size=5)
        pred = [1, 0, 0, 0, 0] # 誤って発火してしまった
        targ = [0, 0, 0, 0, 0]
        
        surprise, signal = minimizer.compute_surprise_signal(pred, targ)
        
        assert surprise[0] == 1 # 誤発火部分がSurpriseとなる
        assert signal == 0.0 # ターゲットがないので重み更新の基準にしない


class TestSpikingJEPA:
    """Spiking JEPA（自己教師あり予測モジュール）のテスト"""

    def test_initialization_and_sparsity(self) -> None:
        """行列ではなく、スパースな辞書ベースの結合が作られているか確認"""
        jepa = SpikingJEPA(context_size=10, target_size=5, hidden_size=8)
        
        assert len(jepa.context_projector) == 8
        assert len(jepa.predictor) == 5
        
        # 全結合(Dense)ではなく、スパース(一部のキーしか持たない)であることを確認
        # 確率的なので稀に全結合になる可能性があるが、通常はサイズ未満になる
        is_sparse = False
        for node_synapses in jepa.context_projector:
            if len(node_synapses) < 10:
                is_sparse = True
                break
        assert is_sparse

    def test_multilingual_message(self) -> None:
        """多言語ステータス機能の確認"""
        jepa = SpikingJEPA(10, 5, 8)
        assert "Spiking JEPA" in jepa.get_status_message("ja")
        assert "predictive coding" in jepa.get_status_message("en")
        assert "rétropropagation" in jepa.get_status_message("fr")

    def test_predictive_learning_convergence(self) -> None:
        """
        [最重要テスト]
        誤った予測（過剰発火）がペナルティシグナルによって抑制され、
        予測誤差（Surprise/エネルギー）が減少していく学習ダイナミクスを検証する。
        """
        random.seed(42)
        jepa = SpikingJEPA(context_size=5, target_size=5, hidden_size=5)
        jepa.learning_rate = 0.8
        jepa.threshold = 0.1
        
        # テスト環境の制御: 確実に探索と刈り込みが起きるように初期結線を仕込む
        for i in range(5):
            jepa.context_projector[i] = {}
            jepa.predictor[i] = {}
            
        # コンテキスト(0,1)入力で、意図的に間違った予測(0,1)が発火するように設定する
        jepa.context_projector[0] = {0: 1.0, 1: 1.0}
        jepa.context_projector[1] = {0: 1.0, 1: 1.0}
        
        jepa.predictor[0] = {0: 1.0, 1: 1.0}
        jepa.predictor[1] = {0: 1.0, 1: 1.0}
        
        # ターゲットはインデックス(2,3)なので、上記の予測(0,1)は間違いとなる
        context_pattern = [1, 1, 0, 0, 0]
        target_pattern = [0, 0, 1, 1, 0]
        
        initial_surprises = []
        final_surprises = []
        
        # 1. 最初の数エポック（学習初期：間違った予測が出るためSurpriseが大きい）
        # ★修正ポイント: 評価中なので learning=False にしなければならない
        for _ in range(5):
            surprise, _ = jepa.step(context_pattern, target_pattern, learning=False)
            initial_surprises.append(sum(surprise))
            # 膜電位の不要な蓄積を防ぐためリセット
            jepa.hidden_potentials = [0.0] * jepa.hidden_size
            jepa.target_potentials = [0.0] * jepa.target_size
            
        # 2. 学習ループ（間違った予測をしたシナプスがLTDで刈り込まれる）
        for _ in range(20):
            jepa.step(context_pattern, target_pattern, learning=True)
            jepa.hidden_potentials = [0.0] * jepa.hidden_size
            jepa.target_potentials = [0.0] * jepa.target_size
            
        # 3. 最後の数エポック（学習後期：無駄な発火が消えSurpriseが減る）
        for _ in range(5):
            surprise, _ = jepa.step(context_pattern, target_pattern, learning=False)
            final_surprises.append(sum(surprise))
            jepa.hidden_potentials = [0.0] * jepa.hidden_size
            jepa.target_potentials = [0.0] * jepa.target_size
            
        avg_initial_surprise = sum(initial_surprises) / len(initial_surprises)
        avg_final_surprise = sum(final_surprises) / len(final_surprises)
        
        # 最初はターゲットにない過剰発火でSurpriseが多いが、
        # 学習による刈り込みで無駄な発火が減り、全体のエラー（エネルギー）が低下する。
        assert avg_final_surprise < avg_initial_surprise

    def test_pruning_mechanism(self) -> None:
        """不要なシナプスが刈り込まれ(Pruning)、メモリを節約しているか確認"""
        random.seed(42)
        jepa = SpikingJEPA(context_size=5, target_size=5, hidden_size=5)
        jepa.learning_rate = 1.0
        jepa.prune_threshold = 0.1
        
        # 意図的に弱いシナプスを作る
        jepa.predictor[0] = {0: 0.05, 1: 2.0} 
        
        # ターゲット発火=0、予測発火=1 の状態を作ってLTD(減衰)を発生させる
        hidden = [1, 0, 0, 0, 0] # 入力0が発火
        pred = [1, 0, 0, 0, 0]   # 出力0が発火
        
        # ネガティブシグナル(-1.0)を与えてSTDP更新を実行
        jepa._update_weights_stdp(hidden, pred, jepa.predictor, signal=-1.0)
        
        # 元々0.05だった入力0への結合は、LTDによって閾値(0.1)を下回り削除されたはず
        assert 0 not in jepa.predictor[0]
        # 元々2.0だった入力1への結合は、ヘテロシナプティックLTDで多少減ったが残っているはず
        assert 1 in jepa.predictor[0]