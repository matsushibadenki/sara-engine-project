from sara_engine.models.spiking_multimodal_hub import SpikingMultimodalHub
import os
_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmarks/benchmark_multimodal_hub.py",
    "//": "ファイルの日本語タイトル: クロスモーダル連想ベンチマーク",
    "//": "ファイルの目的や内容: テキストのスパイクと画像のスパイクを同時に学習させ、片方の入力からもう片方の表現を想起（Fuzzy Recall）できるかをテストする。"
}


def run_benchmark():
    work_dir = "workspace/logs"
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, "multimodal_hub_benchmark.log")

    # テキストと画像の2つのモダリティを持つハブを作成
    hub = SpikingMultimodalHub(modalities=["text", "vision"])

    # --- 学習データ（概念のペア） ---
    # Concept 1: "Apple"
    apple_text = [12, 45, 88]          # 「リンゴ」という単語のスパイク
    apple_vision = [100, 205, 310]     # 「赤くて丸い」という視覚特徴のスパイク

    # Concept 2: "Ocean"
    ocean_text = [15, 66, 92]          # 「海」という単語のスパイク
    ocean_vision = [120, 250, 400]     # 「青くて波打つ」という視覚特徴のスパイク

    print("--- 1. マルチモーダル連想学習 (Hebbian Learning) ---")
    # それぞれのペアを同時に提示して学習させる（数回繰り返して結合を強化）
    for _ in range(3):
        hub.associate({"text": apple_text, "vision": apple_vision})  # type: ignore[attr-defined]
        hub.associate({"text": ocean_text, "vision": ocean_vision})  # type: ignore[attr-defined]

    print("学習が完了しました。")

    print("\n--- 2. クロスモーダル想起テスト ---")
    with open(log_file, "w") as f:
        f.write("Multimodal Hub Recall Log\n")
        f.write("=========================\n\n")

        # テストA: 「リンゴ」のテキストスパイクから、画像のスパイクを想起できるか？
        retrieved_vision_from_apple = hub.retrieve(  # type: ignore[attr-defined]
            source_modality="text",
            source_spikes=apple_text,
            target_modality="vision"
        )
        msg_a = f"[Test A] Text 'Apple' {apple_text} -> Retrieved Vision: {retrieved_vision_from_apple}"
        print(msg_a)
        f.write(msg_a + "\n")

        if set(retrieved_vision_from_apple) == set(apple_vision):
            print("  => 成功: テキストから正しい視覚特徴を完全に想起しました。")

        # テストB: 「海」の画像スパイクの *一部* しか見えなかった場合（曖昧さの許容テスト）
        partial_ocean_vision = [120, 250]  # 400が欠損
        retrieved_text_from_ocean = hub.retrieve(  # type: ignore[attr-defined]
            source_modality="vision",
            source_spikes=partial_ocean_vision,
            target_modality="text",
            threshold=1.0  # 部分的な入力でも想起できるよう閾値を調整
        )
        msg_b = f"[Test B] Partial Vision 'Ocean' {partial_ocean_vision} -> Retrieved Text: {retrieved_text_from_ocean}"
        print(msg_b)
        f.write(msg_b + "\n")

        if set(retrieved_text_from_ocean) == set(ocean_text):
            print("  => 成功: 不完全な視覚情報から、正しいテキストを連想しました（Fuzzy Recall）。")


if __name__ == "__main__":
    run_benchmark()
