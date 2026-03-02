# examples/nlp/demo_direct_wiring.py
# 日本語タイトル: 直接シナプス結線のテストスクリプト
# 目的: コーパスの生成、SNNの直接結線による学習、およびテキスト生成のテストをワークスペース環境下で実行する。
# {
#     "//": "出力ファイル群はプロジェクト肥大化を防ぐためルート階層のworkspaceディレクトリに保存します。"
# }

import os
import sys

# プロジェクトルートのsrcをパスに追加してモジュールを読み込めるようにする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sara_engine.learning.direct_wiring import DirectWiringSNN

def main():
    # Setup workspace paths
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../workspace/logs/direct_wiring'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    corpus_path = os.path.join(workspace_dir, 'sample_corpus.txt')
    model_path = os.path.join(workspace_dir, 'direct_wiring_snn_model.json')
    
    print("[INFO] Setting up the workspace and generating sample corpus...")
    
    # Generate a multi-lingual synthetic corpus for testing
    sample_corpus = """
    SARA Engine is an advanced spiking neural network. It aims for high energy efficiency and biological plausibility.
    人工知能の新しい形として、スパイクニューラルネットワークは非常に重要です。誤差逆伝播法を使わずに学習することが目標です。
    SARA Engine learns text very fast by directly connecting synapses based on token statistics.
    これは多言語に対応しており、英語と日本語が混ざった文章でも機能します。
    """
    
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(sample_corpus * 50) # Multiply to create sufficient statistics
        
    print(f"[INFO] Corpus created at {corpus_path}")
    
    # Read the corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        training_text = f.read()
        
    # Initialize SNN
    snn = DirectWiringSNN(context_window=10)
    
    # Train (Direct Wiring) - No BP, No Matrices, No GPU
    print("[INFO] Starting Direct Synaptic Wiring...")
    snn.train_from_corpus(training_text)
    
    # Save the physical connections
    snn.save_model(model_path)
    
    # Test Generation
    print("\n--- Generation Test 1 (English) ---")
    prompt_en = "SARA Engine is"
    output_en = snn.generate(prompt=prompt_en, max_new_tokens=40)
    print(f"Prompt: '{prompt_en}'\nGenerated: '{output_en}'\n")

    print("--- Generation Test 2 (Japanese) ---")
    prompt_jp = "人工知能の新しい"
    output_jp = snn.generate(prompt=prompt_jp, max_new_tokens=40)
    print(f"Prompt: '{prompt_jp}'\nGenerated: '{output_jp}'\n")

if __name__ == "__main__":
    main()