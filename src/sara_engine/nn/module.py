_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/module.py",
    "//": "ファイルの日本語タイトル: SNN基底モジュール (nn.Module代替)",
    "//": "ファイルの目的や内容: PyTorchのnn.Moduleのように、SNNの層を直感的に組み合わせてネットワークを構築・管理するための基底クラス。動的状態の保存(state_dict)もサポートする。"
}

import collections
from typing import Dict, Any, List, Optional
import copy

class SNNModule:
    # mypyのための型宣言
    _modules: Dict[str, Any]
    _state_vars: List[str]

    def __init__(self) -> None:
        # __setattr__の無限ループを防ぐため辞書に直接初期化
        self.__dict__['_modules'] = collections.OrderedDict()
        self.__dict__['_state_vars'] = []

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, SNNModule):
            self._modules[name] = value
        super().__setattr__(name, value)

    def register_state(self, name: str) -> None:
        """
        内部状態(膜電位、不応期、シナプス重みなど)をstate_dictの保存対象として登録する。
        """
        if name not in self._state_vars:
            self._state_vars.append(name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def reset_state(self) -> None:
        """SNN特有の動的状態(膜電位など)を初期化する"""
        for module in self._modules.values():
            if hasattr(module, 'reset_state'):
                module.reset_state()

    # mypy対応: Optionalを追加
    def state_dict(self, destination: Optional[Dict[str, Any]] = None, prefix: str = '') -> Dict[str, Any]:
        """PyTorchライクにネットワーク全体の状態を辞書として返す"""
        if destination is None:
            destination = collections.OrderedDict()
            
        # ローカルな状態変数を保存
        for var_name in self._state_vars:
            if hasattr(self, var_name):
                val = getattr(self, var_name)
                # 参照渡しを防ぐためディープコピー
                destination[prefix + var_name] = copy.deepcopy(val)
        
        # サブモジュールの状態を再帰的に取得
        for name, module in self._modules.items():
            if hasattr(module, 'state_dict'):
                module.state_dict(destination, prefix + name + '.')
            
        return destination

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = False) -> None:
        """保存された状態を復元する"""
        for var_name in self._state_vars:
            if var_name in state_dict:
                setattr(self, var_name, copy.deepcopy(state_dict[var_name]))
        
        for name, module in self._modules.items():
            # サブモジュール用のキーを抽出
            sub_dict = {k[len(name)+1:]: v for k, v in state_dict.items() if k.startswith(name + '.')}
            if sub_dict and hasattr(module, 'load_state_dict'):
                module.load_state_dict(sub_dict, strict)