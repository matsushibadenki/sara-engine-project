# Directory Path: src/sara_engine/learning/delay_plasticity.py
# English Title: Event-Driven Synaptic Delay Plasticity
# Purpose/Content: O(1)のイベント駆動型ルーティングによるシナプス遅延の自己組織化。ポストニューロン発火時に直近の到着スパイクのみをピンポイントで計算・破棄し、計算量とメモリ使用量を最小化する。多言語対応。

from typing import Dict

class SynapticDelayManager:
    """
    イベント駆動型のSynaptic Delay Plasticity (シナプス遅延の可塑性)。
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
        
        # O(1)アクセス用のイベント駆動キュー: arrivals_by_post[post_id][pre_id] = arrival_time
        self.arrivals_by_post: Dict[int, Dict[int, float]] = {}

    def get_status(self, lang: str = "en") -> str:
        """多言語対応: 遅延マネージャーの状態取得"""
        active_queues = len(self.arrivals_by_post)
        messages = {
            "en": f"DelayManager: Active event queues: {active_queues}, LR: {self.learning_rate}",
            "ja": f"DelayManager: アクティブなイベントキュー数: {active_queues}, 学習率: {self.learning_rate}",
            "fr": f"DelayManager: Files d'attente d'événements actives: {active_queues}, Taux d'apprentissage: {self.learning_rate}"
        }
        return messages.get(lang, messages["en"])

    def get_delay(self, pre_id: int, post_id: int, default_value: float = 5.0) -> float:
        if pre_id not in self.delays:
            self.delays[pre_id] = {}
        return self.delays[pre_id].get(post_id, default_value)

    def record_arrival(self, pre_id: int, post_id: int, arrival_time: float):
        """スパイクがポストニューロンに到着した時刻を、ポストニューロン主導の辞書に記録する"""
        if post_id not in self.arrivals_by_post:
            self.arrivals_by_post[post_id] = {}
        self.arrivals_by_post[post_id][pre_id] = arrival_time

    def update_delays(self, post_id: int, t_post: float):
        """
        ポストニューロンが発火した際、そのニューロンのキューだけを O(1) で処理し、
        処理後は即座にメモリから破棄(Prune)してリソースを解放する。
        """
        if post_id not in self.arrivals_by_post:
            return

        # 該当ポストニューロンに最近届いたスパイクだけをイテレート
        for pre_id, t_arrival in list(self.arrivals_by_post[post_id].items()):
            error = t_post - t_arrival
            
            # 10ms 以内の因果関係を持つスパイクのみ学習対象とする
            if abs(error) < 10.0:
                current_delay = self.get_delay(pre_id, post_id)
                
                # STDP的な非対称カーネルの導入: 早く着きすぎた場合は遅延を伸ばし、遅すぎた場合は縮める
                delta = self.learning_rate * error
                new_delay = current_delay + delta
                
                self.delays[pre_id][post_id] = max(self.min_delay, min(self.max_delay, new_delay))
                
        # 計算が終わった到着履歴は完全に削除し、メモリリークと無駄な走査を防ぐ
        del self.arrivals_by_post[post_id]

    def state_dict(self) -> Dict[str, object]:
        return {"delays": {k: dict(v) for k, v in self.delays.items()}}

    def load_state_dict(self, state: Dict[str, object]):
        if "delays" in state:
            self.delays = {int(k): {int(pk): float(pv) for pk, pv in v.items()} for k, v in state["delays"].items()} # type: ignore