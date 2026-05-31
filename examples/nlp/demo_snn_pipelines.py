_FILE_INFO = {
    "//": "ディレクトリパス: examples/nlp/demo_snn_pipelines.py",
    "//": "ファイルの日本語タイトル: SNNパイプライン・デモ",
    "//": "ファイルの目的や内容: TransformersライクなSNNパイプラインの構築と、STDP学習、テキスト生成のテスト(BPEトークナイザ対応版)。"
}

import os
import json
from sara_engine.auto import (
    AutoTokenizer, 
    AutoSNNModelForSequenceClassification,
    AutoSNNModelForTokenClassification
)
from sara_engine.models.snn_transformer import SNNTransformerConfig, SpikingTransformerModel
from sara_engine.pipelines.text_generation import pipeline as text_pipeline
from sara_engine.pipelines import pipeline

def main():
    # ワークスペースディレクトリの作成 (examples/nlp/../../workspace)
    workspace_dir = os.path.join(os.path.dirname(__file__), "..", "..", "workspace", "pipeline_demo")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # ダミーディレクトリをパスとして使用し、デフォルト設定で新規作成させる
    dummy_model_dir = os.path.join(workspace_dir, "dummy_model")
    
    # ---------------------------------------------------------
    print("--- 1. Testing Text Classification Pipeline (Sentiment Analysis) ---")
    tokenizer = AutoTokenizer.from_pretrained(dummy_model_dir)
    
    # 💡 修正点: トークナイザーに語彙(BPE)を事前学習させる
    cls_texts = [
        "This is an excellent SNN library.", 
        "This error is annoying.", 
        "I hate this bug.",
        "これは素晴らしいSNNライブラリです。",
        "このバグは最悪だ。"
    ]
    tokenizer.train(cls_texts)
    
    cls_model = AutoSNNModelForSequenceClassification.from_pretrained(dummy_model_dir)
    
    # パイプラインの初期化
    cls_pipeline = pipeline("text-classification", model=cls_model, tokenizer=tokenizer, id2label={0: "NEGATIVE", 1: "POSITIVE"})
    
    # 局所学習(STDP)の実行
    epochs = 8
    print(f"Training Text Classification model locally via STDP for {epochs} epochs...")
    for _ in range(epochs):
        cls_pipeline.learn("This is an excellent SNN library.", label_id=1)
        cls_pipeline.learn("I hate this bug.", label_id=0)
        cls_pipeline.learn("これは素晴らしいSNNライブラリです。", label_id=1)
        cls_pipeline.learn("このバグは最悪だ。", label_id=0)
    
    texts_to_classify = [
        "This is an excellent SNN library.", 
        "This error is annoying.", 
        "これは素晴らしいSNNライブラリです。"
    ]
    results = cls_pipeline(texts_to_classify)
    
    for text, res in zip(texts_to_classify, results):
        print(f"Input: {text}")
        print(f"Result: {json.dumps(res, ensure_ascii=False)}")
    print("\n")

    # ---------------------------------------------------------
    print("--- 2. Testing Token Classification Pipeline (NER) ---")
    tok_model = AutoSNNModelForTokenClassification.from_pretrained(dummy_model_dir)
    
    ner_pipeline = pipeline("token-classification", model=tok_model, tokenizer=tokenizer, 
                            id2label={0: "O", 1: "B-ORG", 2: "B-LOC"})
    
    sample_text = "Matsushiba is located in Tokyo."
    test_text = "Where is Matsushiba? Is it in Tokyo?"
    
    # 💡 修正点: NER用のテキストもトークナイザーに学習させる
    tokenizer.train([sample_text, test_text])
    
    # 💡 修正点: BPEトークナイズ後の長さに合わせてラベルを動的に生成する (バイトレベル指定の廃止)
    token_ids = tokenizer.encode(sample_text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    dummy_labels = [0] * len(token_ids)
    
    for i, tok in enumerate(tokens):
        if "Matsushiba" in tok:
            dummy_labels[i] = 1 # B-ORG
        elif "Tokyo" in tok:
            dummy_labels[i] = 2 # B-LOC
        
    print(f"Training Token Classification model locally via STDP for {epochs} epochs...")
    for _ in range(epochs):
        ner_pipeline.learn(sample_text, labels=dummy_labels)
    
    ner_results = ner_pipeline(test_text)
    
    print("NER Extracted Tokens:")
    found_entities = False
    for entity in ner_results:
        if entity["entity"] != "O":
            print(f"  Word: {entity['word']}, Entity: {entity['entity']}, Index: {entity['index']}")
            found_entities = True
            
    if not found_entities:
        print("  (No entities extracted. The model might need more training or a lower firing threshold.)")
    print("\n")

    # ---------------------------------------------------------
    print("--- 3. Testing Text Generation Pipeline (Phase 2: Fuzzy Recall) ---")
    config = SNNTransformerConfig(embed_dim=128, num_layers=2, use_fuzzy=True)
    gen_model = SpikingTransformerModel(config)
    
    training_corpus = "Artificial Intelligence based on Spiking Neural Networks."
    prompt = "Artificial "
    
    # 💡 修正点: DummyUTF8Tokenizer を廃止し、正規のSARA Tokenizerを使用する
    gen_tokenizer = AutoTokenizer.from_pretrained(dummy_model_dir)
    gen_tokenizer.train([training_corpus, prompt])
    
    gen_pipeline = text_pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer)
    
    print(f"Training Text Generation model (One-shot Learning) on: '{training_corpus}'")
    gen_pipeline.learn(training_corpus)
    
    print(f"Prompt: '{prompt}'")
    generated_text = gen_pipeline(prompt, max_new_tokens=15)
    print(f"Generated text: '{generated_text}'")

    # モデルの保存テスト
    save_path = os.path.join(workspace_dir, "saved_pipelines")
    ner_pipeline.save_pretrained(save_path)
    gen_pipeline.save_pretrained(save_path)
    print(f"\nModels saved successfully to {save_path}")

if __name__ == "__main__":
    main()