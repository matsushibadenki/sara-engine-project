_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_rag_persistent.py",
    "//": "タイトル: 永続化対応SNN-RAGデモ",
    "//": "目的: 古いPipelineや存在しないConfig引数を廃止し、最新のSpikingFeatureExtractorとSpikingCausalLMを直接使用する堅牢なコードへ修正する。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.models.spiking_feature_extractor import SpikingFeatureExtractor, SNNFeatureExtractorConfig
from src.sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from src.sara_engine.memory.snn_vector_store import SNNVectorStore
from src.sara_engine.models.spiking_causal_lm import SpikingCausalLM

def main():
    print("=== Persistent SNN RAG System Demo ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'rag_persistent_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    vector_store_dir = os.path.join(workspace_dir, "saved_vector_store")
    tokenizer_path = os.path.join(workspace_dir, "tokenizer.json")
    
    knowledge_base = [
        "The SARA Engine is a bio-inspired AI that does not use matrix multiplications or GPUs.",
        "SARA uses Spiking Neural Networks (SNN) for extreme energy efficiency.",
        "SARA stores knowledge in a biological vector store without FAISS or NumPy."
    ]
    
    # --- 1. トークナイザーの準備 ---
    tokenizer = SpikeTokenizer()
    if os.path.exists(tokenizer_path):
        print("Loading existing Tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training Tokenizer on knowledge base...")
        tokenizer.train(knowledge_base + ["What is the SARA Engine?", "It is"])
        tokenizer.save(tokenizer_path)

    # --- 2. 検索エンジンとベクトルストアの準備 ---
    extractor_config = SNNFeatureExtractorConfig(vocab_size=tokenizer.vocab_size, reservoir_size=512, context_length=32)
    extractor_model = SpikingFeatureExtractor(extractor_config)
    
    if os.path.exists(os.path.join(vector_store_dir, "vector_store.json")):
        print("\n[Database] Found existing SNN Vector Store. Loading from disk...")
        vector_store = SNNVectorStore.from_pretrained(vector_store_dir)
    else:
        print("\n[Database] Building new SNN Vector Store from scratch...")
        vector_store = SNNVectorStore()
        for doc in knowledge_base:
            token_ids = tokenizer.encode(doc)
            emb = extractor_model.forward(token_ids)
            vector_store.add_document(doc, emb)
        vector_store.save_pretrained(vector_store_dir)
    
    print(f"-> Store holds {len(vector_store.documents)} documents ready for search.")

    # --- 3. ユーザーからのクエリと検索 ---
    query = "How does SARA store knowledge?"
    print(f"\n[Search] Querying: '{query}'")
    
    query_emb = extractor_model.forward(tokenizer.encode(query))
    search_results = vector_store.search(query_emb, top_k=1)
    
    retrieved_context = search_results[0][0]
    similarity = search_results[0][1]
    print(f"-> Retrieved Context: '{retrieved_context}' (Similarity: {similarity:.4f})")
    
    # --- 4. 回答生成エンジン(Generator)の構築 ---
    print("\n[Generation] Learning context via STDP...")
    generator_model = SpikingCausalLM(vocab_size=tokenizer.vocab_size, d_model=128, num_layers=2)
    
    epochs = 20
    token_ids = tokenizer.encode(retrieved_context) + [3]
    for _ in range(epochs):
        generator_model.train_step(token_ids, update_backbone=True)
            
    # --- 5. 回答の生成 ---
    prompt_ids = tokenizer.encode("SARA stores knowledge")
    out_ids = generator_model.generate(prompt_ids, max_new_tokens=15)
    output_text = tokenizer.decode(out_ids)
    
    print("-" * 50)
    print(f"Question: {query}")
    print(f"Answer  : {output_text}")
    print("-" * 50)

if __name__ == "__main__":
    main()