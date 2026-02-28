_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_text_classification.py",
    "//": "ファイルの日本語タイトル: SNNテキスト分類デモ",
    "//": "ファイルの目的や内容: UTF-8の文脈長に合わせてエポック数を増加し、完全に推論させる。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.auto import AutoSNNModelForSequenceClassification
from sara_engine.models.spiking_sequence_classifier import SpikingSequenceClassifier, SNNSequenceClassifierConfig
from sara_engine.pipelines import pipeline

class ByteLevelSNNTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))
    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode('utf-8', errors='ignore')

def main():
    print("=== SARA Engine: SNN Text Classification Demonstration ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "snn_cls_demo"))
    os.makedirs(workspace_dir, exist_ok=True)
    model_dir = os.path.join(workspace_dir, "snn_cls_checkpoint")
    
    print("Loading Tokenizer and Model...")
    try:
        model = AutoSNNModelForSequenceClassification.from_pretrained(model_dir)
        print("Loaded existing model.")
    except Exception:
        print("No existing model found. Initializing a new SpikingSequenceClassifier.")
        config = SNNSequenceClassifierConfig(vocab_size=256, num_classes=2)
        model = SpikingSequenceClassifier(config)
        
    tokenizer = ByteLevelSNNTokenizer()
    
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, id2label=id2label)
    
    train_data = [
        ("SARA engine is very efficient and fast.", 1),
        ("誤差逆伝播法を使わないこのAIは素晴らしいですね", 1),
        ("It takes too much energy and runs slow.", 0),
        ("行列演算のせいで動作が重くて使い物になりません", 0)
    ]
    
    print("\n--- Starting Local STDP Learning Phase ---")
    # コンテキスト長(32)に対応するため、エポック数を20に増加
    epochs = 20  
    for epoch in range(epochs):
        for text, label_id in train_data:
            classifier.learn(text, label_id)
    print("Training Complete.")
            
    print("\n--- Text Classification Inference ---")
    test_prompts = [
        "SARA is fast and excellent.",                # 期待: POSITIVE
        "このAIは素晴らしい動作です",               # 期待: POSITIVE
        "It is slow and takes too much energy.",      # 期待: NEGATIVE
        "動作が重くてだめです"                        # 期待: NEGATIVE
    ]
    
    for prompt in test_prompts:
        result = classifier(prompt)[0]
        print(f"Text: '{prompt}'")
        print(f" -> Predicted: {result['label']}\n")
        
    print(f"Saving model state to: {model_dir}")
    classifier.save_pretrained(model_dir)
    
    log_file = os.path.join(workspace_dir, "execution.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SARA SNN Classification Pipeline Execution Log ===\n")
        f.write("Status: SUCCESS\n")
        f.write(f"Model saved at: {model_dir}\n")
        
    print(f"Execution log saved to {log_file}")
    print("=== Demonstration Completed ===")

if __name__ == "__main__":
    main()