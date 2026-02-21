_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/memory/snn_vector_store.py",
    "//": "ファイルの日本語タイトル: SNNベクトルストア",
    "//": "ファイルの目的や内容: SNNから抽出された多次元ベクトルを保存・検索する。Transformersライクな save_pretrained と from_pretrained を実装し、知識ベースの永続化に対応。"
}

import math
import json
import os
from typing import List, Tuple

class SNNVectorStore:
    """
    A lightweight, dependency-free vector store for SNN embeddings.
    Acts as an associative memory module (like a simplified hippocampus) 
    for Retrieval-Augmented Generation (RAG).
    """
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []

    def add_document(self, text: str, embedding: List[float]):
        """Store a document and its corresponding SNN spike embedding."""
        self.documents.append(text)
        self.embeddings.append(embedding)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity without NumPy or matrix operations."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(b * b for b in v2))
        if norm_v1 * norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def search(self, query_embedding: List[float], top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Search for the most similar documents based on the query embedding.
        
        Args:
            query_embedding: The embedding vector of the search query.
            top_k: Number of top results to return.
            
        Returns:
            A list of tuples containing (document_text, similarity_score).
        """
        if not self.documents:
            return []

        results = []
        for doc, emb in zip(self.documents, self.embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            results.append((doc, sim))

        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save_pretrained(self, save_directory: str):
        """Saves the vector store to disk in a dependency-free JSON format."""
        os.makedirs(save_directory, exist_ok=True)
        store_path = os.path.join(save_directory, "vector_store.json")
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Vector store successfully saved to {store_path}")

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Loads the vector store from disk."""
        store_path = os.path.join(save_directory, "vector_store.json")
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Vector store not found at {store_path}")
        
        with open(store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instance = cls()
        instance.documents = data.get("documents", [])
        instance.embeddings = data.get("embeddings", [])
        print(f"Vector store successfully loaded from {store_path}")
        return instance