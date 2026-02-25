_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_multimodal_associative.py",
    "//": "ファイルの日本語タイトル: マルチモーダル連合想起ベンチマーク",
    "//": "ファイルの目的や内容: テキスト（概念）と画像（特徴）が同時に提示された場合に、SNNがその相関を学習し、テキストから画像を想起できるかを確認する。"
}

from sara_engine.nn.multimodal import CrossModalAssociator

def run_multimodal_test():
    print("=== SARA Engine: Multimodal Associative Learning Benchmark ===\n")
    
    # テキスト空間(100次元)と画像空間(512次元)を繋ぐアソシエータ
    associator = CrossModalAssociator(dim_a=100, dim_b=512)
    
    # 概念: "Apple" (Token ID: 50)
    # 画像特徴: [10, 20, 30] (赤い円や果物の形状スパイクと仮定)
    concept_spikes = [50]
    visual_spikes = [10, 20, 30]
    
    print("[*] Learning: Pairing 'Apple' concept with visual features...")
    for _ in range(5):
        associator.forward(spikes_a=concept_spikes, spikes_b=visual_spikes, learning=True)
    
    print("\n[*] Recall Test: What does the model 'see' when it hears 'Apple'?")
    result = associator.forward(spikes_a=concept_spikes, learning=False)
    
    recalled_visuals = result["recall_b"]
    print(f"  Input Concept : {concept_spikes}")
    print(f"  Recalled Visual Spikes: {recalled_visuals}")
    
    success = all(v in recalled_visuals for v in visual_spikes)
    if success:
        print("\n=> SUCCESS: Cross-modal recall achieved! The SNN linked text and vision via temporal coincidence.")
    else:
        print("\n=> FAILED: The association was not formed correctly.")

if __name__ == "__main__":
    run_multimodal_test()