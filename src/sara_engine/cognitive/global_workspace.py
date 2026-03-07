# {
#     "//": "ディレクトリパス: src/sara_engine/cognitive/global_workspace.py",
#     "//": "ファイルの日本語タイトル: グローバル・ワークスペース (意識の場)",
#     "//": "ファイルの目的や内容: 複数の思考候補(アセンブリ)を競争させ、Winner-Take-Allによって1つを「意識」に引き上げる。選択された情報を行動決定やネットワーク全体へのブロードキャストに利用する。"
# }

from ..learning.homeostasis import AdaptiveThresholdHomeostasis

class GlobalWorkspace:
    """
    Global Workspace Theory (GWT) に基づく意識の競争ネットワーク。
    """
    def __init__(
        self,
        num_candidates: int,
        inhibition_factor: float = 0.5,
        winner_threshold: float = 1.5,
        decay: float = 0.9,
    ):
        self.num_candidates = num_candidates
        self.activations = [0.0 for _ in range(num_candidates)]
        self.inhibition_factor = inhibition_factor
        self.decay = decay
        self.winner_threshold = winner_threshold
        self.homeostasis = AdaptiveThresholdHomeostasis(
            target_rate=0.05,
            adaptation_rate=0.25,
            decay=0.92,
            min_threshold=0.0,
            max_threshold=2.5,
            global_weight=0.4,
        )

    def reset(self) -> None:
        self.activations = [0.0 for _ in range(self.num_candidates)]
        self.homeostasis.reset()

    def step(self, candidate_inputs: list[float]) -> int:
        """
        各候補(アセンブリ)からの興奮入力を受け取り、競争させる。
        戻り値: 意識に上った(勝者となった)候補のID。まだ閾値に達していない場合は -1。
        """
        padded_inputs = list(candidate_inputs[:self.num_candidates])
        if len(padded_inputs) < self.num_candidates:
            padded_inputs.extend([0.0] * (self.num_candidates - len(padded_inputs)))
        total_activation = sum(self.activations)
        
        for i in range(self.num_candidates):
            # 相互抑制: 他のすべての候補の活動の総和が、自分への抑制(Inhibition)として働く
            inhibition = (total_activation - self.activations[i]) * self.inhibition_factor
            local_threshold = self.homeostasis.get_threshold(i, 0.0)
            
            # 状態の更新: 前回の余韻 + 新たな興奮 - 他からの抑制
            self.activations[i] = (
                (self.activations[i] * self.decay)
                + padded_inputs[i]
                - inhibition
                - local_threshold
            )
            self.activations[i] = max(0.0, self.activations[i])
            
        # 勝者(Winner)の判定
        winner_id = -1
        max_act = 0.0
        for i, act in enumerate(self.activations):
            if act > max_act:
                max_act = act
                winner_id = i
                
        # 意識への浮上判定 (特定の閾値を超える必要がある)
        has_external_support = winner_id != -1 and padded_inputs[winner_id] > 0.0
        if winner_id != -1 and max_act > self.winner_threshold and has_external_support:
            self.homeostasis.update([winner_id], population_size=self.num_candidates)
            # Winner-Take-All: 勝者以外を強力に抑制し、「1つの思考」だけをクリアにする
            for i in range(self.num_candidates):
                if i != winner_id:
                    self.activations[i] *= 0.1
            return winner_id

        self.homeostasis.update([], population_size=self.num_candidates)
        return -1
