# examples/integrated/integrated_spiking_llm_demo.py
# 日本語タイトル: スパイキングLLM 統合デモ（事前学習＋オンライン対話）
# 目的: 統合されたSpikingLLMを用いて、大規模コーパスの事前学習と、STDPを用いたリアルタイムのオンライン学習・対話を行う。
# {
#     "//": "SpikingLLMの統合API（fit, learn_sequence, generate）の動作をテストします。"
# }

import os
import sys
import time

# プロジェクトルートのsrcをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sara_engine.models.spiking_llm import SpikingLLM

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    corpus_path = os.path.join(project_root, 'data/processed/corpus.txt')
    workspace_dir = os.path.join(project_root, 'workspace/logs/integrated_llm')
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 1. モデルの初期化
    print("[INFO] Initializing SpikingLLM...")
    llm = SpikingLLM(sdr_size=128, vocab_size=65536, context_window=15)

    # 2. 事前学習 (Direct Wiring)
    if os.path.exists(corpus_path):
        print(f"\n[INFO] Loading corpus from {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_data = f.read()
        
        print("[INFO] Starting Direct Wiring Pre-training...")
        start_time = time.time()
        llm.fit(text_data)
        print(f"[INFO] Pre-training completed in {time.time() - start_time:.2f} seconds.")
    else:
        print(f"\n[WARNING] Corpus not found at {corpus_path}. Running without pre-training.")

    # 3. リアルタイム対話とオンライン学習 (STDP)
    print("\n" + "="*50)
    print("SARA Engine Interactive Mode (SpikingLLM Integrated)")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)

    while True:
        try:
            prompt = input("\nUser > ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue

            # a. ユーザーの入力をオンライン学習 (STDP / Fuzzy Recallに登録)
            # これにより、会話の中で教えられた新しい概念をすぐに覚えます
            prompt_tokens = llm.encode_text(prompt)
            llm.learn_sequence(prompt_tokens)

            # b. モデルの推論 (事前知識 + オンライン記憶のハイブリッド)
            print("SARA > ", end="", flush=True)
            output = llm.generate(prompt=prompt, max_new_tokens=60, temperature=0.7)
            print(output)

        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break

    # 4. 学習結果の保存
    save_dir = os.path.join(workspace_dir, "saved_model")
    llm.save_pretrained(save_dir)

if __name__ == "__main__":
    main()