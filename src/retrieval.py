"""
Retrieving top-k chunks from global index given a query.
Embeds the query, then searches the FAISS index.
"""

import numpy as np
from typing import List, Tuple, Optional

from src.corpus import Index, ChunkMeta


def retrieve(
    query: str,
    index: Index,
    top_k: int = 5,
) -> List[Tuple[str, float, ChunkMeta]]:
    """
    Embed a query and retrieve top-k similar chunks from the global index.

    

    Returns:
        List of (chunk_text, similarity_score, metadata) tuples, sorted by score desc.
    """
    # Embed the query
    query_embedding = index.embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    return index.retrieve(query_embedding[0], top_k=top_k)


def retrieve_by_embedding(
    query_embedding: np.ndarray,
    index: Index,
    top_k: int = 5,
) -> List[Tuple[str, float, ChunkMeta]]:
    """
    Retrieve top-k chunks given an already-computed embedding.
    Useful for HyDE 

    Args:
        query_embedding: Pre-computed embedding vector.
        index: The global Index object.
        top_k: Number of chunks to retrieve.

    Returns:
        List of (chunk_text, similarity_score, metadata) tuples.
    """
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    return index.retrieve(query_embedding, top_k=top_k)


def embed_text(text: str, index: Index) -> np.ndarray:
    """
    Embedding a single text using the index's embedder.
    
    """
    embedding = index.embedder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embedding[0]


def embed_texts(texts: List[str], index: Index) -> np.ndarray:
    """
    Embed multiple texts using the index's embedder.
    """
    embeddings = index.embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embeddings
