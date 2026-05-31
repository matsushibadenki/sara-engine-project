_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_nn_module.py",
    "//": "ファイルの日本語タイトル: SNNModule (PyTorch風API) デモ",
    "//": "ファイルの目的や内容: 新しく実装したsara_engine.nnモジュールを用いて、PyTorchライクにSNNを構築し、state_dictの保存と復元をテストする。"
}

import os
import pickle
from sara_engine import nn

# 1. PyTorch風にネットワークを定義
class SimpleSpikingNetwork(nn.SNNModule):
    def __init__(self):
        super().__init__()
        # nn.Sequentialを使って層をスタック
        self.features = nn.Sequential(
            nn.LinearSpike(in_features=64, out_features=128, density=0.2),
            nn.LinearSpike(in_features=128, out_features=32, density=0.3)
        )
        self.classifier = nn.LinearSpike(in_features=32, out_features=10, density=0.5)

    def forward(self, spikes, learning=False):
        x = self.features(spikes, learning=learning)
        out = self.classifier(x, learning=learning)
        return out

def main():
    print("--- Building SNN with PyTorch-like API ---")
    model = SimpleSpikingNetwork()
    
    # ダミーの入力スパイク (発火しているニューロンのインデックス)
    input_spikes = [5, 12, 45, 60]
    
    print(f"Input spikes: {input_spikes}")
    
    # 学習モードでのフォワードパス
    print("\n--- Forward Pass (Learning=True) ---")
    for epoch in range(3):
        out_spikes = model(input_spikes, learning=True)
        print(f"Epoch {epoch+1} Output spikes: {out_spikes}")
        
    # state_dictの取得と保存
    print("\n--- Saving state_dict ---")
    state = model.state_dict()
    
    workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    save_path = os.path.join(workspace_dir, "model_state.pkl")
    
    with open(save_path, "wb") as f:
        pickle.dump(state, f)
    print(f"state_dict saved to {save_path}")
    
    # 新しいモデルインスタンスへのロード
    print("\n--- Loading state_dict to a new model ---")
    new_model = SimpleSpikingNetwork()
    
    # state_dictをロードする前は出力が異なる(ランダム初期化によるスパース結合のため)
    print(f"New model BEFORE load (Output spikes): {new_model(input_spikes)}")
    
    with open(save_path, "rb") as f:
        loaded_state = pickle.load(f)
        
    new_model.load_state_dict(loaded_state)
    
    # ロード後の出力(STDPで学習された元モデルと完全に一致するはず)
    print(f"New model AFTER load  (Output spikes): {new_model(input_spikes)}")

if __name__ == "__main__":
    main()