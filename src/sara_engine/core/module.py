{
    "//": "ディレクトリパス: src/sara_engine/core/module.py",
    "//": "ファイルの日本語タイトル: SARAモジュール基底クラス (nn.Moduleの代替)",
    "//": "ファイルの目的や内容: SNNの各コンポーネントを共通のインターフェースで扱い、状態の保存・読み込み (state_dict) や再帰的なモジュール管理を提供する基底クラス。"
}

import json
from typing import Dict, Any, List

class SaraModule:
    """
    Base class for all SARA neural network modules.
    Provides PyTorch-like API without backpropagation or dense matrix multiplications.
    """
    def __init__(self):
        self._modules: Dict[str, 'SaraModule'] = {}
        self._parameters: Dict[str, Any] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, SaraModule):
            self._modules[name] = value
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], SaraModule):
             # Handle ModuleList-like behavior simply
             for i, mod in enumerate(value):
                 self._modules[f"{name}_{i}"] = mod
        super().__setattr__(name, value)

    def register_parameter(self, name: str, value: Any):
        """Registers a parameter (like weights) that should be saved/loaded."""
        self._parameters[name] = value
        setattr(self, name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)

    def eval(self):
        """Sets the module in evaluation mode."""
        self.train(False)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dictionary containing parameters and child modules' states."""
        state = {}
        for name in self._parameters.keys():
            state[name] = getattr(self, name)
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads parameters and child modules' states from a dictionary."""
        for name in self._parameters.keys():
            if name in state_dict:
                val = state_dict[name]
                self._parameters[name] = val
                setattr(self, name, val)
        for name, module in self._modules.items():
            if name in state_dict:
                module.load_state_dict(state_dict[name])

    def save(self, filepath: str):
        """Saves the module's state to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self, filepath: str):
        """Loads the module's state from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        self.load_state_dict(state)

    def reset_state(self):
        """Resets the dynamic states (like membrane potentials) of the module and its children."""
        for module in self._modules.values():
            module.reset_state()