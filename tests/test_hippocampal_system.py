_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_hippocampal_system.py",
    "//": "タイトル: 皮質-海馬システム VSA(役割バインディング)統合評価スクリプト",
    "//": "目的: VSAの巡回シフトによる役割バインディングを組み込み、文構造に依存したICL推論を検証する。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.core.cortex import CorticalColumn
from sara_engine.memory.hippocampus import CorticoHippocampalSystem
from sara_engine.memory.sdr import SDREncoder
from sara_engine.utils.tokenizer import SaraTokenizer

def run_evaluation():
    test_ltm_file = "test_eval_ltm.pkl"
    vocab_file = "sara_vocab.json"
    
    for f in [test_ltm_file, vocab_file]:
        if os.path.exists(f):
            os.remove(f)

    print("=== SNN/SDR VSA(役割バインディング) メモリシステム評価テスト ===\n")

    print("--- [初期化 1/2] トークナイザーの学習 ---")
    corpus = [
        "SARA は 日本 の 都市 に 住んで います",
        "Python は プログラミング 言語 です",
        "太郎 は リンゴ を 持って います",
        "太郎 は それ を 花子 に 渡し ました",
        "花子 は 果物 が 好き です",
        "リンゴ は 果物 の 一種 です",           
        "太郎 は それ を 大切 に 持って います", 
        "リンゴ は 今 誰 が 持って います か ？",
        # VSAテスト用の新しいコーパス
        "花子 は リンゴ を 食べ ました",
        "太郎 は バナナ を 食べ ました",
        "誰 が リンゴ を 食べ ました か ？"
    ]
    tokenizer = SaraTokenizer(vocab_size=2000, model_path=vocab_file)
    tokenizer.train(corpus)
    print(f"学習完了。語彙サイズ: {len(tokenizer.vocab)}")

    print("\n--- [初期化 2/2] 意味ネットワークとVSAの統合 ---")
    # apply_vsa=True で初期化
    encoder = SDREncoder(input_size=2048, density=0.02, use_tokenizer=True, apply_vsa=True)
    encoder.train_semantic_network(corpus, window_size=4, epochs=3)
    print("SDRビットの連合学習とVSAの準備が完了しました。\n")
    
    cortex = CorticalColumn(
        input_size=2048, 
        hidden_size_per_comp=4096, 
        compartment_names=["Task_ICL"]
    )
    
    system = CorticoHippocampalSystem(
        cortex=cortex, 
        ltm_filepath=test_ltm_file, 
        max_working_memory_size=10
    )

    # ---------------------------------------------------------
    # テスト2: 大規模文脈内学習 (In-Context Learning) VSA構造ベース
    # ---------------------------------------------------------
    print("--- [テスト2] 大規模文脈内学習(ICL) - VSA構造的推論の評価 ---")
    
    system.cortex.reset_short_term_memory()
    system.working_memory.clear()
    system.ltm.clear()

    # VSAの効果を明確にするための前提文脈
    contexts = [
        "リンゴ は 果物 の 一種 です",         # リンゴ = 主語(SUBJECT)
        "花子 は リンゴ を 食べ ました",       # リンゴ = 目的語(OBJECT)
        "太郎 は バナナ を 食べ ました"        # リンゴなし、構造は同じ
    ]
    
    print("前提文脈をワーキングメモリに注入中...")
    for ctx in contexts:
        sdr_ctx = encoder.encode(ctx)
        system.experience_and_memorize(sensory_sdr=sdr_ctx, content=ctx, context="Task_ICL", learning=False)

    # クエリ: リンゴが目的語(OBJECT)として使われている質問
    query = "誰 が リンゴ を 食べ ました か ？"
    query_sdr = encoder.encode(query)
    print(f"\nクエリ: '{query}'")
    
    icl_results = system.in_context_inference(current_sensory_sdr=query_sdr, context="Task_ICL")
    
    print("【ICL推論結果（VSA役割バインディングとマクロ文脈によって引き出された関連記憶）】")
    if icl_results:
        for idx, res in enumerate(icl_results, 1):
            print(f" {idx}. {res['content']} (スコア: {res['score']:.4f})")
    else:
        print(" 関連する記憶が見つかりませんでした。")
        
    for f in [test_ltm_file, vocab_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    print("\nテスト完了。クリーンアップが正常に行われました。")

if __name__ == "__main__":
    run_evaluation()