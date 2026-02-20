_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_hippocampal_system.py",
    "//": "タイトル: 皮質-海馬システム統合評価および時間軸記憶定着テスト",
    "//": "目的: VSAの役割バインディングの検証に加え、新しく実装されたSpatioTemporal STDPによる時間軸を伴うエピソード記憶の定着をテストする。"
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
    vocab_file = "workspace/sara_vocab.json"
    
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
        "花子 は リンゴ を 食べ ました",
        "太郎 は バナナ を 食べ ました",
        "誰 が リンゴ を 食べ ました か ？",
        "A社 の CEO は B氏 です",
        "A社 の CEO が C氏 に 変わり ました",
        "現在 の CEO は 誰 です か ？"
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
    
    # 新しく追加された snn_input_size をエンコーダの input_size(2048) と一致させる
    system = CorticoHippocampalSystem(
        cortex=cortex, 
        ltm_filepath=test_ltm_file, 
        max_working_memory_size=10,
        snn_input_size=2048
    )

    # ---------------------------------------------------------
    # テスト1: 大規模文脈内学習 (In-Context Learning) VSA構造ベース
    # ---------------------------------------------------------
    print("--- [テスト1] 大規模文脈内学習(ICL) - VSA構造的推論の評価 ---")
    
    system.cortex.reset_short_term_memory()
    system.working_memory.clear()
    system.ltm.clear()

    # VSAの効果を明確にするための前提文脈
    contexts_vsa = [
        "リンゴ は 果物 の 一種 です",         
        "花子 は リンゴ を 食べ ました",       
        "太郎 は バナナ を 食べ ました"        
    ]
    
    print("前提文脈をワーキングメモリに注入中...")
    for ctx in contexts_vsa:
        sdr_ctx = encoder.encode(ctx)
        system.experience_and_memorize(sensory_sdr=sdr_ctx, content=ctx, context="Task_ICL", learning=False)

    # クエリ: リンゴが目的語(OBJECT)として使われている質問
    query_vsa = "誰 が リンゴ を 食べ ました か ？"
    query_sdr_vsa = encoder.encode(query_vsa)
    print(f"\nクエリ: '{query_vsa}'")
    
    icl_results = system.in_context_inference(current_sensory_sdr=query_sdr_vsa, context="Task_ICL")
    
    print("【ICL推論結果（VSA役割バインディングとマクロ文脈によって引き出された関連記憶）】")
    if icl_results:
        for idx, res in enumerate(icl_results, 1):
            print(f" {idx}. {res['content']} (スコア: {res['score']:.4f})")
    else:
        print(" 関連する記憶が見つかりませんでした。")
        
    # ---------------------------------------------------------
    # テスト2: STDPと時間軸を伴うエピソード記憶の定着 (Consolidation)
    # ---------------------------------------------------------
    print("\n--- [テスト2] STDPと時間軸を伴うエピソード記憶の定着 ---")
    
    system.cortex.reset_short_term_memory()
    system.working_memory.clear()
    system.ltm.clear()
    
    contexts_temporal = [
        "A社 の CEO は B氏 です",
        "A社 の CEO が C氏 に 変わり ました"
    ]
    
    print("時系列エピソードを順番に経験・記憶中 (STDP学習あり)...")
    for ctx in contexts_temporal:
        sdr_ctx = encoder.encode(ctx)
        # learning=True にすることで、STDPによる SNN step が実行される
        system.experience_and_memorize(sensory_sdr=sdr_ctx, content=ctx, context="Task_ICL", learning=True)
        print(f" -> 記憶完了: {ctx}")
        
    print("\n睡眠・休息フェーズ: consolidate_memories による時系列リプレイと記憶の定着...")
    # 記憶を時間順にリプレイし、STDPを駆動させる
    system.consolidate_memories(context="Task_ICL", replay_count=2)
    print(" -> 定着プロセスがエラーなく完了しました。")

    query_temporal = "現在 の CEO は 誰 です か ？"
    query_sdr_temporal = encoder.encode(query_temporal)
    print(f"\nクエリ: '{query_temporal}'")
    
    # 最近の記憶 (C氏) が Recency Bonus によって上位に来るかを推論
    temporal_results = system.in_context_inference(current_sensory_sdr=query_sdr_temporal, context="Task_ICL")
    
    print("【時系列推論結果（STDPとRecency Bonusによる引き出し）】")
    if temporal_results:
        for idx, res in enumerate(temporal_results, 1):
            print(f" {idx}. {res['content']} (スコア: {res['score']:.4f})")
    else:
        print(" 関連する記憶が見つかりませんでした。")

    # クリーンアップ
    for f in [test_ltm_file, vocab_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    print("\nテスト完了。クリーンアップが正常に行われました。")

if __name__ == "__main__":
    run_evaluation()