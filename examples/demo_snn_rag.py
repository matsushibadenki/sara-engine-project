_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_rag.py",
    "//": "ファイルの日本語タイトル: SNNベースのRAG（検索拡張生成）デモ",
    "//": "ファイルの目的や内容: 知識ベースをベクトル化して検索し、その結果を自己回帰SNNにSTDPで即席学習させてから回答を生成させる。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from sara_engine.models.spiking_feature_extractor import SpikingFeatureExtractor, SNNFeatureExtractorConfig
from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.pipelines import pipeline
from sara_engine.memory.snn_vector_store import SNNVectorStore

def main():
    print("=== SNN Retrieval-Augmented Generation (RAG) System ===")
    print("Integrating Liquid State Machine (Retrieval) and STDP (Generation)...\n")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'rag_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    # --- 1. 知識ベースとトークナイザーの準備 ---
    knowledge_base = [
        "The SARA Engine is a bio-inspired AI that does not use matrix multiplications or GPUs.",
        "SARA uses Spiking Neural Networks (SNN) for extreme energy efficiency.",
        "Traditional LLMs rely on backpropagation, which consumes massive amounts of power.",
        "The feature extractor in SARA uses a Liquid State Machine and Homeostatic Plasticity.",
        "To achieve generation, SARA relies on Spike-Timing-Dependent Plasticity (STDP)."
    ]
    
    tokenizer = SpikeTokenizer()
    print("Training common Tokenizer on knowledge base...")
    tokenizer.train(knowledge_base + ["What is the SARA Engine?", "It is"])
    
    # --- 2. 検索エンジン(Retriever)の構築 ---
    print("\n[Step 1] Building Knowledge Vector Store using LSM Feature Extractor...")
    # 特徴抽出モデルのセットアップ
    extractor_config = SNNFeatureExtractorConfig(embedding_dim=512, leak_rate=0.98, std_decay=0.2, std_recovery=0.05)
    extractor_model = SpikingFeatureExtractor(extractor_config)
    
    # 背景ノイズのダウン・スケーリング（恒常性可塑性）
    extractor_model.habituate([tokenizer.encode(doc) for doc in knowledge_base])
    retriever_pipeline = pipeline("feature-extraction", model=extractor_model, tokenizer=tokenizer)
    
    # ベクトルストアへの格納
    vector_store = SNNVectorStore()
    for doc in knowledge_base:
        emb = retriever_pipeline(doc)
        vector_store.add_document(doc, emb)
    print(f"Successfully stored {len(knowledge_base)} documents in the SNN Vector Store.")

    # --- 3. ユーザーからのクエリと検索 ---
    query = "What is the SARA Engine?"
    print(f"\n[Step 2] Querying: '{query}'")
    
    query_emb = retriever_pipeline(query)
    search_results = vector_store.search(query_emb, top_k=1)
    
    retrieved_context = search_results[0][0]
    similarity = search_results[0][1]
    print(f"-> Retrieved Context: '{retrieved_context}' (Similarity: {similarity:.4f})")
    
    # --- 4. 回答生成エンジン(Generator)の構築と「インコンテキスト学習」 ---
    print("\n[Step 3] Rapidly learning the context via STDP (Biological In-Context Learning)...")
    generator_config = SNNTransformerConfig(vocab_size=tokenizer.vocab_size + 10, embed_dim=128, num_layers=2, ffn_dim=256)
    generator_model = SpikingTransformerModel(generator_config)
    
    # 検索してきた知識を、生成モデルのシナプスにSTDPで急激に焼き付ける（エポックを回して強固に記憶）
    epochs = 20
    for _ in range(epochs):
        # "[EOS] (ID:3)" を付与して文末を学習させる
        token_ids = tokenizer.encode(retrieved_context) + [3]
        generator_model.reset_state()
        for i in range(len(token_ids) - 1):
            generator_model.forward_step(token_ids[i], learning=True, target_id=token_ids[i+1])
            
    # --- 5. 回答の生成 ---
    print("\n[Step 4] Generating Answer...")
    generator_pipeline = pipeline("text-generation", model=generator_model, tokenizer=tokenizer)
    
    # 検索された文脈を元に、プロンプトの続きを自己回帰で出力させる
    prompt = "The SARA Engine is"
    output = generator_pipeline(prompt, max_new_tokens=20)
    
    print("-" * 50)
    print(f"Question: {query}")
    print(f"Answer  : {output[0]['generated_text']}")
    print("-" * 50)

if __name__ == "__main__":
    main()