_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_image_classification.py",
    "//": "ファイルの日本語タイトル: SNN画像分類デモ",
    "//": "ファイルの目的や内容: 8x8のピクセル画像をCNNなしで学習し、'X'か'O'のパターンを認識する。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.auto import AutoSNNModelForImageClassification
from sara_engine.pipelines import pipeline

def ascii_to_pixels(ascii_art: str) -> list[float]:
    """アスキーアートを1次元のピクセル強度(0.0 or 1.0)の配列に変換"""
    pixels = []
    lines = ascii_art.strip().split('\n')
    for line in lines:
        for char in line.replace(" ", ""):
            if char in ["#", "@", "*", "X"]:
                pixels.append(1.0) # スパイク発火
            else:
                pixels.append(0.0) # 発火なし
    return pixels

def main():
    print("=== SARA Engine: Matrix-Free SNN Image Classification ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "snn_vision_demo"))
    os.makedirs(workspace_dir, exist_ok=True)
    model_dir = os.path.join(workspace_dir, "snn_vision_checkpoint")
    
    # モデルの読み込み(新規作成)
    model = AutoSNNModelForImageClassification.from_pretrained(model_dir)
    id2label = {0: "Circle (O)", 1: "Cross (X)"}
    
    # Visionパイプラインのインスタンス化
    classifier = pipeline("image-classification", model=model, id2label=id2label)
    
    # 8x8の学習データ ('O' と 'X')
    img_o_train = """
    ........
    ..####..
    .#....#.
    .#....#.
    .#....#.
    .#....#.
    ..####..
    ........
    """
    
    img_x_train = """
    #......#
    .#....#.
    ..#..#..
    ...##...
    ...##...
    ..#..#..
    .#....#.
    #......#
    """
    
    train_data = [
        (ascii_to_pixels(img_o_train), 0),
        (ascii_to_pixels(img_x_train), 1)
    ]
    
    print("\n[Phase 1]: Learning Spatial Pixel Patterns via STDP...")
    # CNNのような重い学習ではなく、数回のスパイク伝播で形を覚える
    epochs = 10
    for epoch in range(epochs):
        for pixels, label_id in train_data:
            classifier.learn(pixels, label_id)
    print("Training Complete.")
    
    # ノイズを含んだテストデータ（学習時と少し形が違う）
    img_o_test = """
    ........
    ..###...
    .#....#.
    .#....#.
    .#...##.
    .#....#.
    ..###...
    ........
    """
    
    img_x_test = """
    #......#
    .#....#.
    ..#..#..
    ...##...
    ........
    ..#..#..
    .#....#.
    #......#
    """
    
    print("\n[Phase 2]: Inference on Unseen/Noisy Images")
    test_prompts = [
        ("Noisy 'O'", ascii_to_pixels(img_o_test)),
        ("Noisy 'X'", ascii_to_pixels(img_x_test))
    ]
    
    for name, pixels in test_prompts:
        result = classifier(pixels)
        print(f"\nImage: {name}")
        print(f" -> Predicted: {result['label']}")
        
    model.save_pretrained(model_dir)
    print("\n=== Demonstration Completed ===")

if __name__ == "__main__":
    main()