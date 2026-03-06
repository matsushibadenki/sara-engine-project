# ディレクトリパス: src/sara_engine/models/hierarchical_snn.py
# ファイルの日本語タイトル: 階層的スパイク特徴抽出モデル
# ファイルの目的や内容: 文字から単語、文脈へと抽象度を上げる階層構造。既存のTransformerブロックを統合。

from typing import List, Dict, Any, Optional
from ..nn.module import SNNModule
from ..core.transformer import SpikeTransformerBlock


class HierarchicalSNN(SNNModule):
    """
    Phase 2: Hierarchical Feature Extraction.
    Inherits from SNNModule to support state_dict and modular management.
    """

    def __init__(self,
                 layer_configs: List[Dict[str, Any]],
                 use_lif: bool = True):
        super().__init__()
        self.layer_configs = layer_configs
        self.use_lif = use_lif

        # モジュールを順次追加。低次（Edge/Char）から高次（Object/Context）へ。
        for i, config in enumerate(layer_configs):
            block = SpikeTransformerBlock(
                embed_dim=config["embed_dim"],
                hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
                use_lif=use_lif
            )
            setattr(self, f"layer_{i}", block)
            # SNNModuleにサブモジュールとして登録
            self._modules[f"layer_{i}"] = block

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        current_spikes = x_spikes
        # 階層を順番に伝播。各層でスパイクが抽象化される。
        for i in range(len(self.layer_configs)):
            layer = getattr(self, f"layer_{i}")
            current_spikes = layer.forward(current_spikes, learning=learning)
        return current_spikes

    def reset_state(self) -> None:
        """全階層の膜電位状態をリセット"""
        super().reset_state()

    def state_dict(self, destination: Optional[Dict[str, Any]] = None, prefix: str = '') -> Dict[str, Any]:
        """階層構造全体の重みを保存"""
        if destination is None:
            import collections
            destination = collections.OrderedDict()

        destination[prefix + "num_layers"] = len(self.layer_configs)
        return super().state_dict(destination, prefix)
