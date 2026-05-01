import os
import httpx
import logging
from typing import List
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, persist_dir="./chroma_db", ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection_name = "project_knowledge"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.http = httpx.Client(timeout=30.0)
        logger.info("RAG Pipeline initialized with ChromaDB")

    def _get_embedding(self, text: str) -> List[float]:
        response = self.http.post(f"{self.ollama_url}/api/embeddings", json={"model": "phi3:mini", "prompt": text})
        response.raise_for_status()
        return response.json().get("embedding", [])

    def ingest(self, documents: List[str], ids: List[str] = None):
        if not documents: return
        if ids is None:
            ids = [f"doc_{i}" for i in range(self.collection.count(), self.collection.count() + len(documents))]
        embeddings = [self._get_embedding(doc) for doc in documents]
        self.collection.add(documents=documents, embeddings=embeddings, ids=ids)
        logger.info(f"Ingested {len(documents)} documents into ChromaDB")

    def query(self, question: str, n_results: int = 3) -> str:
        query_emb = self._get_embedding(question)
        results = self.collection.query(query_embeddings=[query_emb], n_results=n_results)
        contexts = results.get("documents", [[]])[0]
        if not contexts:
            return ""
        logger.info(f"Retrieved {len(contexts)} context chunks")
        return "\n\n".join(contexts)

    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info("ChromaDB collection cleared")
        except Exception:
            pass

if __name__ == "__main__":
    rag = RAGPipeline()
    rag.clear_collection()
    sample_docs = [
        "Load balancing distributes network traffic across multiple servers to prevent overload.",
        "Round Robin scheduling assigns tasks sequentially to available workers in a distributed system.",
        "ChromaDB is an open-source vector database optimized for storing and querying embeddings.",
        "Fault tolerance ensures system continuity by automatically detecting and recovering from node failures."
    ]
    rag.ingest(sample_docs)
    print("\n Querying: 'How does Round Robin work in distributed systems?'")
    print("Context Retrieved:")
    print(rag.query("How does Round Robin work in distributed systems?"))
