_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_rag_persistent.py",
    "//": "ファイルの日本語タイトル: 永続化対応SNN-RAGデモ",
    "//": "ファイルの目的や内容: SNNVectorStoreの保存・読み込み機能を利用し、既に構築された知識ベースがあればエンコード処理をスキップして爆速でRAG推論を行う実践的なスクリプト。"
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
    print("=== Persistent SNN RAG System Demo ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'rag_persistent_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    vector_store_dir = os.path.join(workspace_dir, "saved_vector_store")
    tokenizer_path = os.path.join(workspace_dir, "tokenizer.json")
    
    # 共通の知識ベース
    knowledge_base = [
        "The SARA Engine is a bio-inspired AI that does not use matrix multiplications or GPUs.",
        "SARA uses Spiking Neural Networks (SNN) for extreme energy efficiency.",
        "SARA stores knowledge in a biological vector store without FAISS or NumPy."
    ]
    
    # --- 1. トークナイザーの準備 (保存されていればロード、なければ学習) ---
    tokenizer = SpikeTokenizer()
    if os.path.exists(tokenizer_path):
        print("Loading existing Tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training Tokenizer on knowledge base...")
        tokenizer.train(knowledge_base + ["What is the SARA Engine?", "It is"])
        tokenizer.save(tokenizer_path)

    # --- 2. 検索エンジンとベクトルストアの準備 ---
    extractor_config = SNNFeatureExtractorConfig(embedding_dim=512, leak_rate=0.98, std_decay=0.2, std_recovery=0.05)
    extractor_model = SpikingFeatureExtractor(extractor_config)
    extractor_model.habituate([tokenizer.encode(doc) for doc in knowledge_base])
    retriever_pipeline = pipeline("feature-extraction", model=extractor_model, tokenizer=tokenizer)
    
    # ★ ここが永続化の実践部分 ★
    if os.path.exists(os.path.join(vector_store_dir, "vector_store.json")):
        print("\n[Database] Found existing SNN Vector Store. Loading from disk...")
        vector_store = SNNVectorStore.from_pretrained(vector_store_dir)
    else:
        print("\n[Database] Building new SNN Vector Store from scratch...")
        vector_store = SNNVectorStore()
        for doc in knowledge_base:
            emb = retriever_pipeline(doc)
            vector_store.add_document(doc, emb)
        
        # 構築したベクトルストアを保存
        vector_store.save_pretrained(vector_store_dir)
    
    print(f"-> Store holds {len(vector_store.documents)} documents ready for search.")

    # --- 3. ユーザーからのクエリと検索 ---
    query = "How does SARA store knowledge?"
    print(f"\n[Search] Querying: '{query}'")
    
    query_emb = retriever_pipeline(query)
    search_results = vector_store.search(query_emb, top_k=1)
    
    retrieved_context = search_results[0][0]
    similarity = search_results[0][1]
    print(f"-> Retrieved Context: '{retrieved_context}' (Similarity: {similarity:.4f})")
    
    # --- 4. 回答生成エンジン(Generator)の構築 ---
    print("\n[Generation] Learning context via STDP...")
    generator_config = SNNTransformerConfig(vocab_size=tokenizer.vocab_size + 10, embed_dim=128, num_layers=2, ffn_dim=256)
    generator_model = SpikingTransformerModel(generator_config)
    
    epochs = 20
    for _ in range(epochs):
        token_ids = tokenizer.encode(retrieved_context) + [3]
        generator_model.reset_state()
        for i in range(len(token_ids) - 1):
            generator_model.forward_step(token_ids[i], learning=True, target_id=token_ids[i+1])
            
    # --- 5. 回答の生成 ---
    generator_pipeline = pipeline("text-generation", model=generator_model, tokenizer=tokenizer)
    prompt = "SARA stores knowledge"
    output = generator_pipeline(prompt, max_new_tokens=15)
    
    print("-" * 50)
    print(f"Question: {query}")
    print(f"Answer  : {output[0]['generated_text']}")
    print("-" * 50)

if __name__ == "__main__":
    main()