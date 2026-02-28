_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_stream_learning.py",
    "//": "ファイルの日本語タイトル: 大規模ストリーム学習デモ",
    "//": "ファイルの目的や内容: 大きなテキストファイルをメモリを消費せずに1バイトずつストリーミングでSTDP学習させる実装。"
}

import sys
import os
import time

# プロジェクトルートの 'src' ディレクトリを正確にパスへ追加
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def create_dummy_dataset(filepath: str, size_mb: float = 0.5):
    """ストリーム学習のテスト用に、数十万文字のダミーテキストファイルを生成する"""
    print(f"Creating dummy dataset ({size_mb} MB)...")
    base_text = (
        "Spiking Neural Networks represent the future of AI. "
        "SNNs consume significantly less energy than traditional ANNs. "
        "スパイキングニューラルネットワークはAIの未来を象徴しています。 "
        "誤差逆伝播法を使わず、局所的なSTDPのみで学習します。\n"
    )
    
    bytes_written = 0
    target_bytes = int(size_mb * 1024 * 1024)
    
    with open(filepath, "w", encoding="utf-8") as f:
        while bytes_written < target_bytes:
            f.write(base_text)
            bytes_written += len(base_text.encode('utf-8'))
            
    print(f"Dataset created at: {filepath}")

def train_stream(model: SpikingTransformerModel, filepath: str):
    """大規模ファイルをメモリに載せず、1バイトずつストリーム学習する"""
    file_size = os.path.getsize(filepath)
    print(f"\n[Phase 1] Starting Stream Learning on {filepath}")
    print(f"File Size: {file_size / 1024 / 1024:.2f} MB")
    
    model.reset_state()
    prev_byte = None
    processed_bytes = 0
    start_time = time.time()
    
    # バイナリモードで読み込むことで、巨大ファイルでもメモリ使用量を極小に抑える
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1)
            if not chunk:
                break
                
            curr_byte = chunk[0]
            if prev_byte is not None:
                # 直前のバイトから現在のバイトへの予測をSTDPでオンライン学習
                model.forward_step(prev_byte, learning=True, target_id=curr_byte)
                
            prev_byte = curr_byte
            processed_bytes += 1
            
            # 進捗を定期的に表示
            if processed_bytes % 50000 == 0:
                elapsed = time.time() - start_time
                speed = processed_bytes / elapsed
                progress = (processed_bytes / file_size) * 100
                print(f"Progress: {progress:.1f}% | Processed: {processed_bytes} bytes | Speed: {speed:.0f} bytes/sec")

        # ファイルの終端に到達したら、EOS (0) を学習させて文を閉じる
        if prev_byte is not None:
            model.forward_step(prev_byte, learning=True, target_id=0)

    print("Stream learning complete!")

def main():
    print("="*60)
    print("SNN Stream Learning & Synapse Serialization Demo")
    print("="*60)

    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace'))
    os.makedirs(workspace_dir, exist_ok=True)
    dataset_path = os.path.join(workspace_dir, "large_corpus.txt")
    model_save_dir = os.path.join(workspace_dir, "snn-transformer-stream")

    # 1. 0.5MB (約50万文字) のダミー大規模コーパスを作成
    if not os.path.exists(dataset_path):
        create_dummy_dataset(dataset_path, size_mb=0.5)

    # 2. モデルの初期化とストリーム学習
    config = SNNTransformerConfig(vocab_size=256, embed_dim=128, num_layers=2, ffn_dim=256)
    model = SpikingTransformerModel(config)
    
    train_stream(model, dataset_path)

    # 3. 学習済みシナプス重みの保存 (Serialization)
    print("\n[Phase 2] Saving Model and Synapses...")
    model.save_pretrained(model_save_dir)

    # 4. 学習済みモデルの読み込みと推論 (Deserialization)
    print("\n[Phase 3] Loading Model and Generating Text...")
    loaded_model = SpikingTransformerModel.from_pretrained(model_save_dir)
    
    output_en = loaded_model.generate("Spiking Neural ", max_length=100)
    print(f"\nResult (EN): {output_en}")
    
    output_jp = loaded_model.generate("誤差逆伝播法を", max_length=100)
    print(f"Result (JP): {output_jp}")

if __name__ == "__main__":
    main()