# [配置するディレクトリのパス]: ./src/snn_models/modular_snn.py
# [ファイルの日本語タイトル]: モジュール型・階層的SNNアーキテクチャ
# [ファイルの目的や内容]:
# ネットワークを「Layer」と「Synapse（Connection）」に分離し、
# 201ノードから数千ノードへのスケールアップ、および多層化を容易にする。

class LIFLayer:
    def __init__(self, n_nodes, label=""):
        self.n = n_nodes
        self.v = [0.0] * n_nodes
        self.spikes = [False] * n_nodes
        self.traces = [0.0] * n_nodes
        # 各層固有のパラメータ設定が可能

class STDPConnection:
    def __init__(self, pre_layer, post_layer, conn_type="all_to_all"):
        self.pre = pre_layer
        self.post = post_layer
        # 行列を使わず、(pre_idx, post_idx) のペアと重みをリストで管理
        self.synapses = [] 
        self._initialize_weights(conn_type)

    def update_weights(self):
        # 発火したニューロンに関連するインデックスだけをループ
        # STDPロジックをここに集約
        pass