"""
Building  global corpus from dataset; embedding all chunks; building/saving/loading FAISS index.

"""

import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from tqdm import tqdm

from src.data_loader import load_examples, get_passages_for_retrieval


@dataclass
class ChunkMeta:
    """Metadata for a single chunk in the corpus."""
    source_row_id: str  # interaction_id of the source example
    page_url: str
    page_name: str
    domain: str
    question_type: str


@dataclass
class Index:
    """In-memory FAISS index with chunk texts and metadata."""
    faiss_index: faiss.Index
    chunks: List[str]
    metadata: List[ChunkMeta]
    embedder: object  # SentenceTransformer model

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, ChunkMeta]]:
        """
        We will retrieve top-k chunks by cosine similarity.

        Args:
            query_embedding: Normalized embedding vector (1, dim) or (dim,).
            top_k: Number of results to return.

        Returns:
            List of (chunk_text, similarity_score, metadata) tuples, sorted by score desc.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensuring normalized for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(query_embedding)

        top_k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score), self.metadata[idx]))
        return results


def build_index(
    dataset_path: Optional[str] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    limit: Optional[int] = None,
    batch_size: int = 256,
) -> Index:
    """
    Building the global corpus and FAISS index from the CRAG dataset.

   
         First we will iterate all examples via data_loader.
         Then we will collect every non-empty page_snippet → flat list of chunks with metadata.
         Then we will deduplicate exact-duplicate snippets.
         Then we will embed all chunks with SentenceTransformer (batched, normalized).
         Finally we will build FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity).

    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {embedding_model_name}...")
    embedder = SentenceTransformer(embedding_model_name)

    print("Collecting chunks from dataset...")
    chunks = []
    metadata_list = []
    seen = set()

    for example in tqdm(load_examples(path=dataset_path, limit=limit), desc="Loading examples"):
        interaction_id = example.get("interaction_id", "")
        domain = example.get("domain", "")
        question_type = example.get("question_type", "")

        for sr in example["search_results"]:
            snippet = (sr.get("page_snippet") or "").strip()
            if not snippet:
                continue

            # Deduplicate by exact text
            if snippet in seen:
                continue
            seen.add(snippet)

            chunks.append(snippet)
            metadata_list.append(ChunkMeta(
                source_row_id=interaction_id,
                page_url=sr.get("page_url", ""),
                page_name=sr.get("page_name", ""),
                domain=domain,
                question_type=question_type,
            ))

    print(f"Corpus size: {len(chunks)} unique chunks (from {len(seen)} deduplicated)")

    if len(chunks) == 0:
        raise ValueError("No chunks found in dataset. Check dataset path and contents.")

    print(f"Embedding {len(chunks)} chunks (batch_size={batch_size})...")
    embeddings = embedder.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    print(f"Index built: {faiss_index.ntotal} vectors, dim={dim}")
    return Index(
        faiss_index=faiss_index,
        chunks=chunks,
        metadata=metadata_list,
        embedder=embedder,
    )


def save_index(index: Index, path: str) -> None:
    """
    Save the FAISS index and corpus data to disk.
    """
    dir_path = Path(path).parent
    dir_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index.faiss_index, f"{path}.faiss")

    with open(f"{path}.pkl", "wb") as f:
        pickle.dump({
            "chunks": index.chunks,
            "metadata": index.metadata,
        }, f)

    print(f"Index saved to {path}.faiss and {path}.pkl")


def load_index(path: str, embedding_model_name: str = "all-MiniLM-L6-v2") -> Index:
    """
    Load a previously saved index from disk.
    """
    from sentence_transformers import SentenceTransformer

    faiss_path = f"{path}.faiss"
    pkl_path = f"{path}.pkl"

    if not Path(faiss_path).exists() or not Path(pkl_path).exists():
        raise FileNotFoundError(f"Index files not found at {path}. Run build_index first.")

    print(f"Loading FAISS index from {faiss_path}...")
    faiss_index = faiss.read_index(faiss_path)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loading embedding model: {embedding_model_name}...")
    embedder = SentenceTransformer(embedding_model_name)

    index = Index(
        faiss_index=faiss_index,
        chunks=data["chunks"],
        metadata=data["metadata"],
        embedder=embedder,
    )
    print(f"Index loaded: {faiss_index.ntotal} vectors, {len(index.chunks)} chunks")
    return index


def get_or_build_index(config: dict) -> Index:
    """
    Convenience: load index from disk if available, otherwise build and save.
    """
    index_path = config.get("index_path", "index/crag_index")
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")

    if Path(f"{index_path}.faiss").exists() and Path(f"{index_path}.pkl").exists():
        return load_index(index_path, embedding_model)
    else:
        index = build_index(
            dataset_path=config.get("dataset_path"),
            embedding_model_name=embedding_model,
        )
        save_index(index, index_path)
        return index


if __name__ == "__main__":
    # Quick test: build index from first 5 examples
    print("Building index from first 5 examples...")
    idx = build_index(limit=5)
    print(f"\nChunks: {len(idx.chunks)}")
    print(f"Index size: {idx.faiss_index.ntotal}")
    print(f"First chunk preview: {idx.chunks[0][:100]}...")

    # Test retrieval
    query = "Who directed Inception?"
    qemb = idx.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    results = idx.retrieve(qemb[0], top_k=3)
    print(f"\nQuery: {query}")
    for i, (text, score, meta) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f} | {text[:80]}...")
