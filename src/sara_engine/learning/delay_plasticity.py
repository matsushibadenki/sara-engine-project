# {
#     "//": "ディレクトリパス: src/sara_engine/learning/delay_plasticity.py",
#     "//": "ファイルの日本語タイトル: シナプス遅延可塑性マネージャー",
#     "//": "ファイルの目的や内容: シナプス伝達遅延（Delay）の自己組織化を管理する。スパイクの到着タイミングをポストニューロンの発火に同期させるように遅延時間を調整し、時系列パターンの学習能力を強化する。"
# }

import math
from typing import Dict, List, Any

class SynapticDelayManager:
    """
    Synaptic Delay Plasticity (シナプス遅延の可塑性)
    スパイクの到着時間(t_pre + delay)をポストニューロンの発火時間(t_post)に近づける。
    """

    def __init__(
        self,
        min_delay: float = 1.0,
        max_delay: float = 20.0,
        learning_rate: float = 0.05
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.learning_rate = learning_rate

        # 各シナプスの現在の遅延時間: delays[pre_id][post_id]
        self.delays: Dict[int, Dict[int, float]] = {}
        # 直近の到着記録を保持: arrivals[pre_id][post_id] = arrival_time
        self.last_arrivals: Dict[int, Dict[int, float]] = {}

    def get_delay(self, pre_id: int, post_id: int, default_value: float = 5.0) -> float:
        """特定のシナプスの遅延時間を取得する"""
        if pre_id not in self.delays:
            self.delays[pre_id] = {}
        return self.delays[pre_id].get(post_id, default_value)

    def record_arrival(self, pre_id: int, post_id: int, arrival_time: float):
        """スパイクがポストニューロンに『届いた』時刻を記録する"""
        if pre_id not in self.last_arrivals:
            self.last_arrivals[pre_id] = {}
        self.last_arrivals[pre_id][post_id] = arrival_time

    def update_delays(self, post_id: int, t_post: float):
        """
        ポストニューロンが発火した際に呼び出す。
        そのニューロンに届いた直近のスパイクの遅延を調整する。
        """
        for pre_id, posts in self.last_arrivals.items():
            if post_id in posts:
                t_arrival = posts[post_id]
                
                # 到着と発火の誤差 (ε = t_post - t_arrival)
                # ε > 0: スパイクが早く着きすぎた -> 遅延を増やす
                # ε < 0: スパイクが遅すぎた -> 遅延を減らす
                error = t_post - t_arrival
                
                # 指標的に 10ms 以内の誤差のみを学習対象とする
                if abs(error) < 10.0:
                    current_delay = self.get_delay(pre_id, post_id)
                    
                    # 遅延の更新
                    delta = self.learning_rate * error
                    new_delay = current_delay + delta
                    
                    # 範囲内にクリップ
                    self.delays[pre_id][post_id] = max(self.min_delay, min(self.max_delay, new_delay))

    def state_dict(self) -> Dict[str, object]:
        return {"delays": {k: dict(v) for k, v in self.delays.items()}}

    def load_state_dict(self, state: Dict[str, object]):
        if "delays" in state:
            self.delays = {int(k): {int(pk): float(pv) for pk, pv in v.items()} for k, v in state["delays"].items()} # type: ignore