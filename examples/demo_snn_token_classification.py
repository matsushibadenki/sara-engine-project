_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_token_classification.py",
    "//": "ファイルの日本語タイトル: SNNトークン分類(NER)デモ",
    "//": "ファイルの目的や内容: プレフィックス依存を断ち切った厳格なNER推論を検証する。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.auto import AutoSNNModelForTokenClassification, AutoTokenizer
from sara_engine.pipelines import pipeline

def main():
    print("=== SARA Engine: Matrix-Free SNN Named Entity Recognition (NER) ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "snn_ner_demo"))
    os.makedirs(workspace_dir, exist_ok=True)
    model_dir = os.path.join(workspace_dir, "snn_ner_checkpoint")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoSNNModelForTokenClassification.from_pretrained(model_dir)
    
    id2label = {0: "O", 1: "PER", 2: "LOC", 3: "ORG"}
    ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, id2label=id2label)
    
    train_data = [
        ("田中は東京に行きました。", [("田中", 1), ("東京", 2)]),
        ("SARAエンジンは画期的なAIです。", [("SARA", 3)]),
        ("山田は大阪のSARA支社にいます。", [("山田", 1), ("大阪", 2), ("SARA", 3)])
    ]
    
    print("\n[Phase 1]: Local STDP Learning of Entities (Byte-level)...")
    epochs = 40  
    for epoch in range(epochs):
        for text, labels in train_data:
            ner_pipeline.learn(text, labels)
    print("Training Complete.")
    
    print("\n[Phase 2]: Extracting Entities from Unseen Sentences")
    test_prompts = [
        "田中は大阪に行きました。",       # 期待: 田中(PER), 大阪(LOC)
        "山田が開発したSARAはすごいです。" # 期待: 山田(PER), SARA(ORG)
    ]
    
    for prompt in test_prompts:
        results = ner_pipeline(prompt)
        print(f"\nText: '{prompt}'")
        if not results:
            print(" -> No entities found.")
        else:
            for entity in results:
                print(f" -> Found [{entity['entity']}]: {entity['word']}")
                
    model.save_pretrained(model_dir)
    print("\n=== Demonstration Completed ===")

if __name__ == "__main__":
    main()