# {
#     "//": "ディレクトリパス: src/sara_engine/cognitive/global_workspace.py",
#     "//": "ファイルの日本語タイトル: グローバル・ワークスペース (意識の場)",
#     "//": "ファイルの目的や内容: 複数の思考候補(アセンブリ)を競争させ、Winner-Take-Allによって1つを「意識」に引き上げる。選択された情報を行動決定やネットワーク全体へのブロードキャストに利用する。"
# }

class GlobalWorkspace:
    """
    Global Workspace Theory (GWT) に基づく意識の競争ネットワーク。
    """
    def __init__(self, num_candidates: int, inhibition_factor: float = 0.5):
        self.num_candidates = num_candidates
        self.activations = [0.0 for _ in range(num_candidates)]
        self.inhibition_factor = inhibition_factor
        self.decay = 0.9

    def step(self, candidate_inputs: list[float]) -> int:
        """
        各候補(アセンブリ)からの興奮入力を受け取り、競争させる。
        戻り値: 意識に上った(勝者となった)候補のID。まだ閾値に達していない場合は -1。
        """
        total_activation = sum(self.activations)
        
        for i in range(self.num_candidates):
            # 相互抑制: 他のすべての候補の活動の総和が、自分への抑制(Inhibition)として働く
            inhibition = (total_activation - self.activations[i]) * self.inhibition_factor
            
            # 状態の更新: 前回の余韻 + 新たな興奮 - 他からの抑制
            self.activations[i] = (self.activations[i] * self.decay) + candidate_inputs[i] - inhibition
            self.activations[i] = max(0.0, self.activations[i])
            
        # 勝者(Winner)の判定
        winner_id = -1
        max_act = 0.0
        for i, act in enumerate(self.activations):
            if act > max_act:
                max_act = act
                winner_id = i
                
        # 意識への浮上判定 (特定の閾値を超える必要がある)
        if winner_id != -1 and max_act > 1.5:
            # Winner-Take-All: 勝者以外を強力に抑制し、「1つの思考」だけをクリアにする
            for i in range(self.num_candidates):
                if i != winner_id:
                    self.activations[i] *= 0.1
            return winner_id
            
        return -1