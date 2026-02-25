_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_pipelines.py",
    "//": "ファイルの日本語タイトル: SNNパイプライン・デモ",
    "//": "ファイルの目的や内容: TransformersライクなSNNパイプラインの構築と、STDP学習の定着(複数エポック)テスト。"
}

import os
import json
from sara_engine.auto import (
    AutoTokenizer, 
    AutoSNNModelForSequenceClassification,
    AutoSNNModelForTokenClassification
)
from sara_engine.pipelines import pipeline

def main():
    # ワークスペースディレクトリの作成
    workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace", "pipeline_demo")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # ダミーディレクトリをパスとして使用し、デフォルト設定で新規作成させる
    dummy_model_dir = os.path.join(workspace_dir, "dummy_model")
    
    print("--- 1. Testing Text Classification Pipeline (Sentiment Analysis) ---")
    tokenizer = AutoTokenizer.from_pretrained(dummy_model_dir)
    cls_model = AutoSNNModelForSequenceClassification.from_pretrained(dummy_model_dir)
    
    # パイプラインの初期化
    # 0: Negative, 1: Positive
    cls_pipeline = pipeline("text-classification", model=cls_model, tokenizer=tokenizer, id2label={0: "NEGATIVE", 1: "POSITIVE"})
    
    # 局所学習(STDP)の実行
    # SNNのシナプス重みを閾値以上に強化するため、複数回(エポック)反復提示する
    epochs = 8
    print(f"Training Text Classification model locally via STDP for {epochs} epochs...")
    for _ in range(epochs):
        cls_pipeline.learn("This is an excellent SNN library.", label_id=1)
        cls_pipeline.learn("I hate this bug.", label_id=0)
        cls_pipeline.learn("これは素晴らしいSNNライブラリです。", label_id=1)
        cls_pipeline.learn("このバグは最悪だ。", label_id=0)
    
    # 推論の実行 (学習した文脈に似た入力)
    texts_to_classify = [
        "This is an excellent SNN library.", 
        "This error is annoying.", # 未学習だがどうなるか
        "これは素晴らしいSNNライブラリです。"
    ]
    results = cls_pipeline(texts_to_classify)
    
    for text, res in zip(texts_to_classify, results):
        print(f"Input: {text}")
        print(f"Result: {json.dumps(res, ensure_ascii=False)}")
    print("\n")


    print("--- 2. Testing Token Classification Pipeline (NER) ---")
    tok_model = AutoSNNModelForTokenClassification.from_pretrained(dummy_model_dir)
    
    # 0: O, 1: B-ORG, 2: B-LOC
    ner_pipeline = pipeline("token-classification", model=tok_model, tokenizer=tokenizer, 
                            id2label={0: "O", 1: "B-ORG", 2: "B-LOC"})
    
    sample_text = "Matsushiba is located in Tokyo."
    
    # バイトレベルトークナイザーのため、文字列のバイト表現とラベルを同期させる
    # "Matsushiba" (10 bytes), " is located in " (15 bytes), "Tokyo" (5 bytes), "." (1 byte)
    # Total: 31 bytes
    dummy_labels = [0] * len(sample_text.encode('utf-8'))
    # Matsushiba (最初の10バイト) を B-ORG (1) に
    for i in range(0, 10):
        dummy_labels[i] = 1
    # Tokyo (25〜29バイト目) を B-LOC (2) に
    for i in range(25, 30):
        dummy_labels[i] = 2
        
    print(f"Training Token Classification model locally via STDP for {epochs} epochs...")
    for _ in range(epochs):
        ner_pipeline.learn(sample_text, labels=dummy_labels)
    
    # 推論の実行 (学習した文脈を含む入力)
    ner_results = ner_pipeline("Where is Matsushiba? Is it in Tokyo?")
    
    print("NER Extracted Tokens:")
    found_entities = False
    for entity in ner_results:
        # 'O' (Other) 以外のものを表示
        if entity["entity"] != "O":
            print(f"  Word: {entity['word']}, Entity: {entity['entity']}, Index: {entity['index']}")
            found_entities = True
            
    if not found_entities:
        print("  (No entities extracted. The model might need more training or a lower firing threshold.)")
            
    # モデルの保存テスト
    save_path = os.path.join(workspace_dir, "saved_pipelines")
    ner_pipeline.save_pretrained(save_path)
    print(f"\nModel saved successfully to {save_path}")

if __name__ == "__main__":
    main()