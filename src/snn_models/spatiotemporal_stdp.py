# [配置するディレクトリのパス]: ./src/snn_models/hierarchical_stdp.py
# [ファイルの日本語タイトル]: 2層感覚野（低次・高次）を備えた階層的STDPモデル
# [ファイルの目的や内容]: 
# 感覚野を2段階に分け、CNNのように情報の抽象化段階を設ける。
# すべての結合層でSTDP学習を行い、報酬なしでの自己組織化を試行する。
# 行列演算、GPU、誤差逆伝播法を一切使用しない純粋なPython処理。

import math
import random
from typing import List

class HierarchicalSNN:
    def __init__(self, n_in=201, n_low=100, n_high=40, n_ctx=20, dt=1.0):
        self.n = {"in": n_in, "low": n_low, "high": n_high, "ctx": n_ctx}
        self.dt = dt
        
        # 膜電位・発火フラグ・トレースの初期化
        self.v = {k: [0.0 for _ in range(v)] for k, v in self.n.items() if k != "in"}
        self.spikes = {k: [False for _ in range(v)] for k, v in self.n.items()}
        self.traces = {k: [0.0 for _ in range(v)] for k, v in self.n.items()}
        
        # パラメータ
        self.v_thresh = 0.45
        self.v_reset = 0.0
        self.tau_m = 10.0
        self.tau_trace = 20.0
        self.A_plus = 0.01
        self.A_minus = 0.012 # LTDを少し強めにし、活動の飽和を防ぐ
        self.w_max = 2.0
        
        # 重みと結合リストの構築 (行列を使わないスパース管理)
        # 1. In -> Low (局所受容野)
        self.W_in_low, self.conn_in_low = self._build_sparse_conn(n_in, n_low, rf_size=10)
        # 2. Low -> High (やや広い受容野)
        self.W_low_high, self.conn_low_high = self._build_sparse_conn(n_low, n_high, rf_size=15)
        # 3. High -> Ctx (全結合)
        self.W_high_ctx, self.conn_high_ctx = self._build_full_conn(n_high, n_ctx)
        # 4. Ctx -> Ctx (再帰結合)
        self.W_ctx_ctx, self.conn_ctx_ctx = self._build_full_conn(n_ctx, n_ctx, self_conn=False)

    def _build_sparse_conn(self, n_pre, n_post, rf_size):
        weights = [[0.0 for _ in range(n_post)] for _ in range(n_pre)]
        conn_pre_to_post: List[List[int]] = [[] for _ in range(n_pre)]
        for j in range(n_post):
            start = int(j * (n_pre - rf_size) / max(1, n_post - 1))
            for i in range(start, min(n_pre, start + rf_size)):
                weights[i][j] = random.uniform(0.1, 0.4)
                conn_pre_to_post[i].append(j)
        return weights, conn_pre_to_post

    def _build_full_conn(self, n_pre, n_post, self_conn=True):
        weights = [[random.uniform(0.1, 0.3) for _ in range(n_post)] for _ in range(n_pre)]
        conn_pre_to_post: List[List[int]] = [[] for _ in range(n_pre)]
        for i in range(n_pre):
            for j in range(n_post):
                if not self_conn and i == j:
                    weights[i][j] = 0.0
                    continue
                conn_pre_to_post[i].append(j)
        return weights, conn_pre_to_post

    def step(self, heat_data):
        # 1. Input Layer
        for i in range(self.n["in"]):
            p_fire = min(1.0, heat_data[i] * 2.0)
            self.spikes["in"][i] = random.random() < p_fire
            if self.spikes["in"][i]: self.traces["in"][i] += 1.0

        # 2. Layer Processing (Low -> High -> Ctx)
        self._process_layer("low", self.spikes["in"], self.W_in_low, self.conn_in_low)
        self._process_layer("high", self.spikes["low"], self.W_low_high, self.conn_low_high)
        
        # Ctx層のみ再帰入力を加算
        self._process_layer("ctx", self.spikes["high"], self.W_high_ctx, self.conn_high_ctx, recurrent=True)

        # 3. STDP Learning (全結合ペアに対して更新)
        self._apply_stdp(self.traces["in"], self.spikes["low"], self.W_in_low, self.conn_in_low, self.traces["low"], self.spikes["in"])
        self._apply_stdp(self.traces["low"], self.spikes["high"], self.W_low_high, self.conn_low_high, self.traces["high"], self.spikes["low"])
        self._apply_stdp(self.traces["high"], self.spikes["ctx"], self.W_high_ctx, self.conn_high_ctx, self.traces["ctx"], self.spikes["high"])
        self._apply_stdp(self.traces["ctx"], self.spikes["ctx"], self.W_ctx_ctx, self.conn_ctx_ctx, self.traces["ctx"], self.spikes["ctx"], decay=True)

        # 4. Decay Traces
        df = math.exp(-self.dt / self.tau_trace)
        for k in self.traces:
            for i in range(len(self.traces[k])): self.traces[k][i] *= df

    def _process_layer(self, layer_key, pre_spikes, weights, conn_map, recurrent=False):
        n_post = self.n[layer_key]
        n_pre = len(pre_spikes)
        I_syn = [0.0 for _ in range(n_post)]
        
        # 順伝播
        for i in range(n_pre):
            if pre_spikes[i]:
                for j in conn_map[i]:
                    I_syn[j] += weights[i][j]
        
        # 再帰結合 (Ctx層のみ)
        if recurrent:
            for i in range(n_post):
                if self.spikes["ctx"][i]:
                    for j in self.conn_ctx_ctx[i]:
                        I_syn[j] += self.W_ctx_ctx[i][j]

        # LIF更新
        for j in range(n_post):
            self.v[layer_key][j] += (-(self.v[layer_key][j]) + I_syn[j]) * (self.dt / self.tau_m)
            if self.v[layer_key][j] >= self.v_thresh:
                self.v[layer_key][j] = self.v_reset
                self.spikes[layer_key][j] = True
                self.traces[layer_key][j] += 1.0
            else:
                self.spikes[layer_key][j] = False

    def _apply_stdp(self, pre_traces, post_spikes, weights, conn_map, post_traces, pre_spikes, decay=False):
        # LTP: ポストが光った時、プレの履歴を見る
        for j in range(len(post_spikes)):
            if post_spikes[j]:
                for i in range(len(pre_spikes)):
                    # 接続がある場合のみ（スパース対応）
                    if j in conn_map[i]:
                        weights[i][j] += self.A_plus * pre_traces[i]
                        if decay: weights[i][j] -= 0.0001 * weights[i][j] # 恒常性
                        if weights[i][j] > self.w_max: weights[i][j] = self.w_max
        
        # LTD: プレが光った時、ポストの履歴を見る
        for i in range(len(pre_spikes)):
            if pre_spikes[i]:
                for j in conn_map[i]:
                    weights[i][j] -= self.A_minus * post_traces[j]
                    if weights[i][j] < 0.0: weights[i][j] = 0.0