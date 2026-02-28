# [配置するディレクトリのパス]: ./examples/test_spike_dataloader.py
# [ファイルの日本語タイトル]: スパイクデータローダーとロガーの動作テスト
# [ファイルの目的や内容]: 新規実装したTextSpikeDataset, SpikeDataLoader, SaraBoardLoggerが連動して正しくイベント駆動型データを処理・記録できるかを確認するテストスクリプト。
_FILE_INFO = {
    "//": "ディレクトリパス: examples/test_spike_dataloader.py",
    "//": "ファイルの日本語タイトル: スパイクデータローダーとロガーの動作テスト",
    "//": "ファイルの目的や内容: 新規実装したTextSpikeDataset, SpikeDataLoader, SaraBoardLoggerが連動して正しくイベント駆動型データを処理・記録できるかを確認するテストスクリプト。"
}

import os
import shutil
from sara_engine.utils.data.dataset import TextSpikeDataset # type: ignore
from sara_engine.utils.data.dataloader import SpikeDataLoader # type: ignore
from sara_engine.utils.sara_board import SaraBoardLogger # type: ignore

def main():
    print("=== SARA Engine: Spike DataLoader & Logger Test ===\n")
    
    # テスト用のクリーンなログディレクトリを準備
    test_log_dir = "workspace/test_logs"
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)
        
    # 1. ダミーデータの作成 (例: 何らかの文章がトークン化されたIDリストと仮定)
    dummy_token_ids = [105, 22, 45, 99, 12, 0]
    print(f"[*] Input Token IDs: {dummy_token_ids}")
    
    # 2. データセットの初期化 (1トークンあたり10タイムステップ間隔で発火させると設定)
    print("[*] Initializing TextSpikeDataset...")
    dataset = TextSpikeDataset(text_ids=dummy_token_ids, time_step_per_token=10)
    
    # 3. データローダーの初期化
    print("[*] Initializing SpikeDataLoader...")
    dataloader = SpikeDataLoader(dataset=dataset)
    
    # 4. ロガーの初期化
    print("[*] Initializing SaraBoardLogger...")
    logger = SaraBoardLogger(log_dir=test_log_dir)
    
    # 5. データローダーからスパイクストリームを取得し、ロガーに記録
    print("\n[*] Processing Spike Stream & Logging...")
    for timestamp, spikes in dataloader:
        print(f"  -> Generated Event | Time: {timestamp:03d}, Spikes: {spikes}")
        
        # 取得したスパイクをロガーに記録 (ダミーのレイヤー名を指定)
        for spike_id in spikes:
            logger.log_spike(timestamp=timestamp, layer_name="InputCortex", neuron_id=spike_id)
            
            # STDPによるシナプス荷重変化の記録テスト (ダミーデータでテスト)
            # 例として、発火したニューロンから常にID:200への結合が強化されたと仮定
            if timestamp > 0:
                dummy_weight = 0.5 + (timestamp / 100.0)
                logger.log_weight_change(timestamp=timestamp, pre_id=spike_id, post_id=200, new_weight=dummy_weight)

    # 6. ロガーから履歴を取得して正しく保存されたか検証
    print("\n[*] Retrieving Spike History from Logger...")
    spike_history = logger.get_spike_history()
    
    success = True
    if len(spike_history) == len(dummy_token_ids):
        print("  -> OK: Logged spike count matches input tokens.")
    else:
        print(f"  -> ERROR: Expected {len(dummy_token_ids)} logs, got {len(spike_history)}")
        success = False

    for record in spike_history:
        print(f"  Recorded Log: {record}")
        
    print("\n[*] Retrieving Weight History from File...")
    weight_log_path = os.path.join(test_log_dir, "weights.jsonl")
    if os.path.exists(weight_log_path):
        with open(weight_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"  -> OK: Found {len(lines)} weight change records.")
            if lines:
                print(f"  Sample Weight Log: {lines[0].strip()}")
    else:
        print("  -> ERROR: Weight log file not found.")
        success = False

    if success:
        print("\n=== All Tests Completed Successfully! ===")
    else:
        print("\n=== Test Failed. Please check the errors above. ===")

if __name__ == "__main__":
    main()