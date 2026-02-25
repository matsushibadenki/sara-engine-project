_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/sequential.py",
    "//": "ファイルの日本語タイトル: シーケンシャル・コンテナ",
    "//": "ファイルの目的や内容: PyTorchのnn.Sequentialに相当し、複数のSNNモジュールを直列に繋いで順伝播させるコンテナクラス。"
}

import inspect
from .module import SNNModule

class Sequential(SNNModule):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.__setattr__(str(idx), module)

    def reset_state(self):
        for module in self._modules.values():
            module.reset_state()

    def forward(self, spikes, learning=False):
        out = spikes
        for name, module in self._modules.items():
            # モジュールがlearning引数を受け取るか判定
            sig = inspect.signature(module.forward)
            if 'learning' in sig.parameters:
                out = module(out, learning=learning)
            else:
                out = module(out)
        return out