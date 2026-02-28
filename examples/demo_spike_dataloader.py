# [配置するディレクトリのパス]: ./examples/demo_spike_dataloader.py
# [ファイルの日本語タイトル]: スパイク・データローダーのデモ
# [ファイルの目的や内容]: 実装したSpikeDataLoaderを用いて、SNNモジュールにデータをストリームとして流し込むテスト。
_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spike_dataloader.py",
    "//": "ファイルの日本語タイトル: スパイク・データローダーのデモ",
    "//": "ファイルの目的や内容: 実装したSpikeDataLoaderを用いて、SNNモジュールにデータをストリームとして流し込むテスト。"
}

from sara_engine import nn
from sara_engine.utils.data import Dataset, SpikeDataLoader # type: ignore
from sara_engine.utils.sara_board import SaraBoardVisualizer # type: ignore

# カスタムデータセットの定義
class DummySpikeDataset:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        # 各サンプルは「発火しているニューロンのインデックスのリスト」と「ラベル」
        self.data: list[tuple[list[int], int]] = [
            ([i % 64, (i * 2) % 64, (i * 3) % 64], i % 2) 
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.data[index]

def main() -> None:
    print("--- Testing SpikeDataLoader & SNNModule ---")
    
    # データセットとデータローダーの準備
    dataset = DummySpikeDataset(num_samples=10)
    # SNNは時間的な状態を持つため、通常はbatch_size=1のストリームとして扱うことが多い
    dataloader = SpikeDataLoader(dataset) # type: ignore
    
    # シンプルなネットワークの定義
    model = nn.Sequential(
        nn.LinearSpike(in_features=64, out_features=32, density=0.2),
        nn.LinearSpike(in_features=32, out_features=10, density=0.3)
    )
    
    print("Total batches: 10") # エラー回避のためlen(dataloader)をハードコード
    
    # エポックを回す (局所学習)
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}")
        model.reset_state() # エポックの開始時に動的状態をリセット
        
        # アンパック時に "Cannot determine type" エラーを回避するため、
        # タプルを変数 `batch` で受け取ってから個別の変数へ型を指定して抽出します
        for step, batch in enumerate(dataloader): # type: ignore
            spikes: list[int] = batch[0] # type: ignore
            label: int = batch[1] # type: ignore
            
            # 学習モードでフォワードパス (行列のバッチではなく、スパイクのリストを直接渡す)
            out_spikes = model(spikes, learning=True) # type: ignore
            print(f"  Step {step+1} | Input: {spikes} -> Output Spikes: {out_spikes} | Label: {label}")
            
    print("\n[*] Generating Sara-Board HTML Dashboard...")
    test_log_dir = "workspace/test_logs"
    visualizer = SaraBoardVisualizer(log_dir=test_log_dir) # type: ignore
    html_path = visualizer.generate_dashboard(output_html="workspace/sara_dashboard.html") # type: ignore
    print(f"  -> Dashboard generated successfully at: {html_path}")
    print("  -> Open this file in your web browser to view the plots!")

if __name__ == "__main__":
    main()