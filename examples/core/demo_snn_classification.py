# examples/demo_snn_classification.py
_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_classification.py",
    "//": "ファイルの日本語タイトル: SNNテキスト分類パイプラインのデモ",
    "//": "ファイルの目的や内容: 学習順序によるRecency Bias（最近効果）を防ぐため、エポックごとにデータをシャッフルして学習させる。"
}

import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.models.spiking_sequence_classifier import SpikingSequenceClassifier, SNNSequenceClassifierConfig
from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.pipelines import pipeline

def main():
    print("=== SNN Text Classification Pipeline Demo ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'classification_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 1. トークナイザーの準備
    tokenizer = SpikeTokenizer()
    training_data = [
        ("I love this amazing biological network.", 1),
        ("The system is extremely fast and efficient.", 1),
        ("I hate the old slow algorithms.", 0),
        ("The system uses a terrible and bad approach.", 0),
        ("素晴らしいシステム、最高のアプローチです。", 1),
        ("最悪のシステム、非常に遅いです。", 0)
    ]
    
    print("\nTraining Tokenizer...")
    tokenizer.train([text for text, label in training_data])
    tokenizer_path = os.path.join(workspace_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # 2. モデルの初期化と学習（Error-Driven STDP + Data Shuffling）
    config = SNNSequenceClassifierConfig(vocab_size=tokenizer.vocab_size + 10, num_classes=2)
    model = SpikingSequenceClassifier(config)
    
    print("\nTraining Classification SNN with Error-Driven STDP...")
    epochs = 20
    for epoch in range(epochs):
        correct = 0
        # 生物の記憶定着プロセスのように、毎回異なる順序で経験(データ)を再生し、特定のシナプスへの偏りを防ぐ
        random.shuffle(training_data)
        
        for text, label in training_data:
            token_ids = tokenizer.encode(text)
            predicted = model.forward(token_ids, learning=True, target_class=label)
            if predicted == label:
                correct += 1
                
        acc = correct / len(training_data) * 100
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs} - Accuracy: {acc:5.1f}%")
            
    model_dir = os.path.join(workspace_dir, "saved_classifier")
    model.save_pretrained(model_dir)
    print(f"\nClassifier successfully saved to {model_dir}\n")

    # 3. パイプラインでの推論テスト
    print("Initializing Text Classification Pipeline...")
    classifier = pipeline("text-classification", model=model_dir, tokenizer=tokenizer, id2label={0: "NEGATIVE", 1: "POSITIVE"})
    
    test_prompts = [
        "I love this efficient network.",
        "The old algorithms are terrible.",
        "素晴らしいシステムです。"
    ]
    
    print("-" * 50)
    for prompt in test_prompts:
        output = classifier(prompt)
        print(f"Input: '{prompt}' -> Prediction: {output[0]['label']}")
    print("-" * 50)

if __name__ == "__main__":
    main()