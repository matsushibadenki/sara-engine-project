_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spike_dataloader.py",
    "//": "ファイルの日本語タイトル: スパイク・データローダーのデモ",
    "//": "ファイルの目的や内容: 実装したSpikeDataLoaderを用いて、SNNモジュールにデータをストリームとして流し込むテスト。"
}

from sara_engine import nn
from sara_engine.utils.data import Dataset, SpikeDataLoader

# カスタムデータセットの定義
class DummySpikeDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        # 各サンプルは「発火しているニューロンのインデックスのリスト」と「ラベル」
        self.data = [
            ([i % 64, (i * 2) % 64, (i * 3) % 64], i % 2) 
            for i in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index]

def main():
    print("--- Testing SpikeDataLoader & SNNModule ---")
    
    # データセットとデータローダーの準備
    dataset = DummySpikeDataset(num_samples=10)
    # SNNは時間的な状態を持つため、通常はbatch_size=1のストリームとして扱うことが多い
    dataloader = SpikeDataLoader(dataset, batch_size=1, shuffle=True)
    
    # シンプルなネットワークの定義
    model = nn.Sequential(
        nn.LinearSpike(in_features=64, out_features=32, density=0.2),
        nn.LinearSpike(in_features=32, out_features=10, density=0.3)
    )
    
    print(f"Total batches: {len(dataloader)}")
    
    # エポックを回す (局所学習)
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}")
        model.reset_state() # エポックの開始時に動的状態をリセット
        
        for step, (spikes, label) in enumerate(dataloader):
            # 学習モードでフォワードパス (行列のバッチではなく、スパイクのリストを直接渡す)
            out_spikes = model(spikes, learning=True)
            print(f"  Step {step+1} | Input: {spikes} -> Output Spikes: {out_spikes} | Label: {label}")

if __name__ == "__main__":
    main()