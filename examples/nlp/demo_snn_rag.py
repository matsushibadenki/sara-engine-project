_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_rag.py",
    "//": "ファイルの日本語タイトル: SNNベース RAG (検索拡張生成) デモ",
    "//": "ファイルの目的や内容: 検索したテキストを反復学習させ、進捗をプログレス出力しながらSNNに続きを生成させる。"
}

import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.auto import AutoSNNModelForFeatureExtraction, AutoModelForCausalSNN, AutoTokenizer
from sara_engine.pipelines import pipeline

def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def main():
    print("=== SARA Engine: 100% Matrix-Free SNN RAG Demonstration ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "snn_rag_demo"))
    os.makedirs(workspace_dir, exist_ok=True)
    feat_model_dir = os.path.join(workspace_dir, "snn_feat_checkpoint")
    gen_model_dir = os.path.join(workspace_dir, "snn_gen_checkpoint")
    
    print("Loading Models and Tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(feat_model_dir)
    
    feat_model = AutoSNNModelForFeatureExtraction.from_pretrained(feat_model_dir)
    retriever = pipeline("feature-extraction", model=feat_model, tokenizer=tokenizer)
    
    gen_model = AutoModelForCausalSNN.from_pretrained(gen_model_dir)
    generator = pipeline("text-generation", model=gen_model, tokenizer=tokenizer)
    
    knowledge_base = [
        "SARAエンジンは、行列演算と誤差逆伝播法を完全に排除したAIです。",
        "日本の首都は東京であり、人口が非常に密集しています。",
        "STDP(スパイクタイミング依存可塑性)を用いることで局所学習を実現します。",
        "今日の天気は晴れで、湿度が低く過ごしやすい一日です。"
    ]
    
    print("\n[Phase 1]: Encoding Knowledge Base into Spiking SDRs...")
    kb_vectors = []
    for doc in knowledge_base:
        kb_vectors.append(retriever(doc))
        print(f"  Encoded: {doc[:15]}...")
        
    query = "SARAエンジンの学習方法は何ですか？"
    print(f"\n[Phase 2]: User Query -> '{query}'")
    query_vector = retriever(query)
    
    scored_docs = []
    for i, doc_vec in enumerate(kb_vectors):
        score = compute_cosine_similarity(query_vector, doc_vec)
        scored_docs.append((score, knowledge_base[i]))
        print(f"  Similarity with Doc {i+1}: {score:.4f}")
        
    # スコアで降順ソートし、上位2つのドキュメントを取得する
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_k = 2
    best_docs = [doc for score, doc in scored_docs[:top_k]]
    combined_context = " ".join(best_docs)
            
    print(f"\n=> Top Retrieved Documents for Context:")
    for i, doc in enumerate(best_docs):
        print(f"  Rank {i+1}: '{doc}'")
    
    print("\n[Phase 3]: Generation with Retrieved Context...")
    # SNNが文脈を理解できるよう、質問と回答のペアとしてSTDP学習させる
    train_text = f"質問: {query}\n回答: {combined_context}"
    
    epochs = 30
    print(f"  -> Local STDP Learning... ({epochs} epochs)")
    for epoch in range(epochs):
        generator.learn(train_text)
        # プログレス出力を行い、フリーズしていないことを視覚的に確認する
        sys.stdout.write(f"\r    Epoch {epoch+1}/{epochs} completed.")
        sys.stdout.flush()
    print("\n  -> Learning Finished.")
    
    # 質問を与えて続きを生成させる
    # 日本語はトークン数を多く消費するため、max_lengthを多めに確保する
    prompt = f"質問: {query}\n回答: "
    generated_text = generator(prompt, max_length=512)
    print(f"\nPrompt: '{prompt}'")
    print(f"SNN Output: '{generated_text}'")
    
    feat_model.save_pretrained(feat_model_dir)
    gen_model.save_pretrained(gen_model_dir)
    
    log_file = os.path.join(workspace_dir, "execution.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SARA SNN RAG Demonstration Log ===\n")
        f.write("Status: SUCCESS\n")
        f.write(f"Retrieved Context: {combined_context}\n")
        f.write(f"Generated: {generated_text}\n")
    print("\n=== Demonstration Completed ===")

if __name__ == "__main__":
    main()