from src.sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from src.sara_engine.memory.snn_vector_store import SNNVectorStore
from src.sara_engine.models.spiking_causal_lm import SpikingCausalLM
from src.sara_engine.models.spiking_feature_extractor import SpikingFeatureExtractor, SNNFeatureExtractorConfig
import math
import sys
import os
_FILE_INFO = {
    "//": "ディレクトリパス: examples/integrated_snn_rag_demo.py",
    "//": "ファイルの日本語タイトル: 統合SNN-RAGシステムデモ",
    "//": "ファイルの目的や内容: SNNを用いたベクトル検索、ナレッジベースの永続化、および検索結果をコンテキストとした回答生成（STDP学習）を一つのワークフローで実証する。"
}


# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """行列演算ライブラリ(NumPy等)を使わず、標準の演算のみでコサイン類似度を計算"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def main():
    print("=== SARA Engine: Integrated SNN-RAG System Demonstration ===\n")

    workspace_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../workspace/integrated_rag'))
    os.makedirs(workspace_dir, exist_ok=True)
    vector_store_dir = os.path.join(workspace_dir, "vector_store")

    # 1. 知識ベースの準備
    knowledge_base = [
        "SARA Engine is a bio-inspired AI that does not use matrix multiplications.",
        "It achieves learning through STDP (Spike-Timing-Dependent Plasticity).",
        "The system is designed for extreme energy efficiency on edge devices.",
        "日本の首都は東京であり、SARAの設計チームの一部はそこに拠点を置いています。"
    ]

    # 2. トークナイザーと特徴抽出器の初期化
    tokenizer = SpikeTokenizer()
    tokenizer.train(knowledge_base + ["What is SARA?", "Learning method"])

    extractor_config = SNNFeatureExtractorConfig(
        vocab_size=tokenizer.vocab_size, reservoir_size=512)
    extractor = SpikingFeatureExtractor(extractor_config)
    print(f"[1] Components initialized. Vocab size: {tokenizer.vocab_size}")

    # 3. ベクトルストアの構築と永続化
    if os.path.exists(os.path.join(vector_store_dir, "vector_store.json")):
        print("[2] Loading existing SNN Vector Store...")
        vector_store = SNNVectorStore.from_pretrained(vector_store_dir)
    else:
        print("[2] Building new SNN Vector Store...")
        vector_store = SNNVectorStore()
        for doc in knowledge_base:
            token_ids = tokenizer.encode(doc)
            # SNNを用いた分散表現（SDR）の抽出
            embedding = extractor.forward(token_ids)
            vector_store.add_document(doc, embedding)
        vector_store.save_pretrained(vector_store_dir)
    print(f"    Store holds {len(vector_store.documents)} documents.")

    # 4. 検索 (Retrieval)
    query = "How does SARA learn?"
    print(f"\n[3] Search Phase: '{query}'")
    query_ids = tokenizer.encode(query)
    query_emb = extractor.forward(query_ids)

    # ベクトル検索の実行
    results = vector_store.search(query_emb, top_k=1)
    retrieved_context, similarity = results[0]
    print(
        f"    Retrieved: '{retrieved_context}' (Similarity: {similarity:.4f})")

    # 5. 生成 (Generation via Contextual STDP)
    print("\n[4] Generation Phase: Learning retrieved context...")
    generator = SpikingCausalLM(
        vocab_size=tokenizer.vocab_size, embed_dim=256, num_layers=2)

    # 検索された知識を文脈としてSTDP学習
    context_tokens = tokenizer.encode(retrieved_context) + [3]  # EOS
    epochs = 25
    for epoch in range(epochs):
        generator.train_step(context_tokens)
        if (epoch + 1) % 5 == 0:
            sys.stdout.write(f"\r    Epoch {epoch+1}/{epochs} training...")
            sys.stdout.flush()
    print("\n    Contextual learning complete.")

    # プロンプトに基づいて回答を生成
    prompt = "SARA learns using"
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = generator.generate(prompt_ids, max_new_tokens=10)

    print("-" * 50)
    print(f"Query    : {query}")
    print(f"Prompt   : {prompt}")
    print(f"Response : {tokenizer.decode(generated_ids)}")
    print("-" * 50)


if __name__ == "__main__":
    main()
