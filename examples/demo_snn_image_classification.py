_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_image_classification.py",
    "//": "ファイルの日本語タイトル: SNN画像分類パイプラインのデモ",
    "//": "ファイルの目的や内容: 外部ライブラリを使わず、5x5ピクセルの簡易画像をレートコーディングとSTDPで学習させ、パターン認識を行うテスト。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.models.spiking_image_classifier import SpikingImageClassifier, SNNImageClassifierConfig
from sara_engine.pipelines import pipeline

def print_image(img_2d):
    for row in img_2d:
        print("".join(["██" if pixel > 0.5 else "  " for pixel in row]))

def flatten(img_2d):
    return [pixel for row in img_2d for pixel in row]

def main():
    print("=== SNN Image Classification (Multimodal) Demo ===")
    print("Using Retinal Rate Coding and STDP for biological pattern recognition.\n")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'image_classification_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 5x5 Synthetic Images for demonstration (1.0 = white/spike, 0.0 = black/rest)
    image_X = [
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
    ]
    
    image_O = [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
    ]
    
    training_data = [
        (flatten(image_X), 0), # 0: Cross
        (flatten(image_O), 1)  # 1: Circle
    ]
    
    # 1. Initialize SNN Vision Model (5x5 = 25 input pixels)
    config = SNNImageClassifierConfig(input_dim=25, num_classes=2, time_steps=15, leak_rate=0.9)
    model = SpikingImageClassifier(config)
    
    # 2. Train with Error-Driven STDP
    print("Training SNN Vision Model with Error-Driven STDP...")
    epochs = 30
    for epoch in range(epochs):
        correct = 0
        for pixels, label in training_data:
            predicted = model.forward(pixels, learning=True, target_class=label)
            if predicted == label:
                correct += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs} - Accuracy: {correct / len(training_data) * 100:.1f}%")
            
    model_dir = os.path.join(workspace_dir, "saved_vision_model")
    model.save_pretrained(model_dir)
    print(f"\nVision Model saved to {model_dir}\n")

    # 3. Initialize Pipeline
    print("Initializing Image Classification Pipeline...")
    vision_pipeline = pipeline("image-classification", model=model_dir, id2label={0: "Cross (X)", 1: "Circle (O)"})
    
    # 4. Inference Test (We pass 2D arrays directly, pipeline flattens it)
    print("\n--- Pattern Recognition Test ---")
    
    test_cases = [("Pattern 1", image_X), ("Pattern 2", image_O)]
    for name, img in test_cases:
        print(f"\n[{name}] Input Image:")
        print_image(img)
        
        output = vision_pipeline(img)
        print(f"-> SNN Prediction: {output[0]['label']}")

if __name__ == "__main__":
    main()