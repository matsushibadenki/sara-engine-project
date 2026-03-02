# examples/nlp/demo_train_large_corpus.py
# 日本語タイトル: 大規模コーパスを用いたSNN直接結線学習テスト
# 目的: 実際のプロジェクトデータ（corpus.txt）を読み込み、Rustコアによる超高速One-Shot学習のストレステストと語彙の拡大を行う。
# {
#     "//": "プロジェクト内の実際のテキストデータを読み込みます。ファイルがない場合は警告を出して終了します。"
# }

import os
import sys
import time

# プロジェクトルートのsrcをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sara_engine.learning.direct_wiring import DirectWiringSNN

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    
    # 読み込む大規模コーパスのパス（既存のプロジェクトデータを使用）
    corpus_path = os.path.join(project_root, 'data/processed/corpus.txt')
    
    # ワークスペースパスの設定
    workspace_dir = os.path.join(project_root, 'workspace/logs/direct_wiring')
    os.makedirs(workspace_dir, exist_ok=True)
    model_path = os.path.join(workspace_dir, 'large_snn_model.json')

    if not os.path.exists(corpus_path):
        print(f"[ERROR] コーパスファイルが見つかりません: {corpus_path}")
        print("[INFO] 別のテキストファイルがある場合は、corpus_path を書き換えてください。")
        return

    print(f"[INFO] Loading large corpus from: {corpus_path}")
    
    # テキストの読み込み（最大サイズを制限する場合はここでスライス）
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        text_data = f.read()
        
    print(f"[INFO] Corpus loaded. Length: {len(text_data)} characters.")

    # SNNの初期化（コンテキストウィンドウを広げて、より長い文脈を捉える）
    snn = DirectWiringSNN(context_window=15)
    
    # 学習の実行と時間計測
    print("[INFO] Starting ultra-fast direct wiring via Rust core...")
    start_time = time.time()
    
    snn.train_from_corpus(text_data)
    
    elapsed_time = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed_time:.2f} seconds!")
    
    # モデルの保存
    snn.save_model(model_path)
    
    print("\n[INFO] モデルの保存が完了しました。")
    print(f"[INFO] 引き続き `python examples/nlp/demo_direct_wiring_chat.py` を実行して、大規模な語彙を持ったSNNと対話してみてください！")

if __name__ == "__main__":
    main()