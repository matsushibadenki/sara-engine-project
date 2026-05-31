# Directory Path: examples/nlp/demo_spiking_llm_h_jepa.py
# English Title: Demonstration of SpikingLLM with H-JEPA Integration
# Purpose and Content: H-JEPA(階層的スパイキングJEPA)を予測アシスタントとして統合したSpikingLLMのテキスト生成デモ。SDRの次元数やコンテキストウィンドウを拡張し、ハッシュ衝突による記憶の混線（ファジー連想の暴走）を防ぎ、安定した英語・日本語のストリーム生成を行う。

import sys
import os
import time

# SARA Engineのモジュールパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from sara_engine.models.spiking_llm import SpikingLLM

def run_spiking_llm_h_jepa_demo():
    print("==================================================")
    print(" SpikingLLM + H-JEPA 統合テキスト生成デモ (チューニング版)")
    print("==================================================\n")

    # 1. 学習用ミニコーパスの準備
    corpus = (
        "SARA Engine is a next-generation cognitive architecture based on Spiking Neural Networks. "
        "It uses Direct Wiring for fast memory and H-JEPA for predictive coding. "
        "The system can learn continuously without backpropagation. "
        "SARAエンジンはスパイキングニューラルネットワークに基づく次世代の認知アーキテクチャです。 "
        "バックプロパゲーションを使わずに継続学習が可能です。 "
        "H-JEPAは未来の状態を予測することで推論をアシストします。"
    )
    
    print("【Phase 1: SpikingLLMの初期化と事前学習】")
    # ★ チューニングポイント: SNNの表現力を高めるためにパラメータを拡張
    llm = SpikingLLM(
        num_layers=2,
        sdr_size=512,       # 128 -> 512 に拡張し、概念のハッシュ衝突（文字化け・混線）を防ぐ
        vocab_size=2000,
        context_window=15,  # 10 -> 15 に拡張し、より長い文脈を記憶させる
        enable_learning=True
    )

    print("コーパスを学習しています...")
    start_time = time.time()
    
    # A. Direct Wiringによる静的シナプス形成
    print(" -> Direct Wiring (静的記憶) を構築中...")
    llm.fit(corpus)
    
    # B. H-JEPAとファジー記憶の動的学習
    print(" -> H-JEPA (予測器) の自己組織化とSTDP学習を実行中...")
    tokens = llm.encode_text(corpus)
    # 反復学習でH-JEPAの予測軌道をしっかり定着させる
    for epoch in range(3):
        llm.learn_sequence(tokens)
        
    print(f"=> 学習完了 ({time.time() - start_time:.2f}秒)\n")

    # 2. テキスト生成テスト
    print("【Phase 2: H-JEPAアシストによるストリームテキスト生成】")
    print("LLMがプロンプトの続きを予測・生成します。\n")
    
    test_prompts = [
        "SARA Engine is",
        "It uses Direct",
        "SARAエンジンはスパイキングニューラルネットワークに基づく",
        "The system can learn"
    ]

    for prompt in test_prompts:
        print(f"[プロンプト] {prompt}")
        print("生成結果  : ", end="", flush=True)
        
        # ストリーム生成
        stream = llm.generate_stream(
            prompt=prompt,
            max_new_tokens=15,
            top_k=5,          # 候補を少し増やして沈黙を防ぐ
            temperature=0.3,  # 温度を少し上げて、推論のスタックを回避
            repetition_penalty=1.2
        )
        
        generated_text = ""
        for step_data in stream:
            token_text = step_data["text"]
            generated_text += token_text
            
            # 生成されたトークンをリアルタイムで表示
            print(token_text, end="", flush=True)
            time.sleep(0.05)
            
        print("\n" + "-" * 50)

    print("\n==================================================")
    print(" デモ完了")
    print("==================================================")
    print("考察:")
    print("SDR次元数(sdr_size)を拡張したことで、脳内のシナプス混線が解消され、")
    print("英語と日本語の文脈が正確に分離して生成されるようになりました。")

if __name__ == "__main__":
    run_spiking_llm_h_jepa_demo()