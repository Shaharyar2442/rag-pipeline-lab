"""
Graph RAG: graph-augmented retrieval over the corpus
"""

import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from src.retrieval import retrieve, embed_text
from src.generation import generate_answer, generate_text
from src.corpus import Index, ChunkMeta



def _extract_entities_simple(text: str) -> Set[str]:
    """
    Extracting named entities using simple heuristics:
    """
    entities = set()

    # Extracting capitalized phrases (2+ words or single capitalized words > 2 chars)
    cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    for match in re.finditer(cap_pattern, text):
        entity = match.group(1).strip()
        if len(entity) > 2 and entity.lower() not in _STOP_ENTITIES:
            entities.add(entity.lower())

    # Also extracting single capitalized words that are likely entities
    single_cap = r'\b([A-Z][a-z]{2,})\b'
    for match in re.finditer(single_cap, text):
        word = match.group(1).lower()
        if word not in _STOP_ENTITIES and len(word) > 3:
            entities.add(word)

    return entities


def _extract_entities_spacy(text: str) -> Set[str]:
    """Extract named entities using spaCy NER."""
    try:
        import spacy
        if not hasattr(_extract_entities_spacy, '_nlp'):
            try:
                _extract_entities_spacy._nlp = spacy.load("en_core_web_sm")
            except OSError:
                return _extract_entities_simple(text)
        
        doc = _extract_entities_spacy._nlp(text[:5000])  # Limit text length for speed
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", 
                              "FAC", "PRODUCT", "NORP", "DATE"):
                name = ent.text.strip().lower()
                if len(name) > 2:
                    entities.add(name)
        return entities
    except ImportError:
        return _extract_entities_simple(text)


# These are some words that look like entities but aren't useful
_STOP_ENTITIES = {
    "the", "this", "that", "these", "those", "there", "here",
    "what", "which", "who", "whom", "when", "where", "how", "why",
    "has", "have", "had", "was", "were", "been", "being",
    "are", "can", "could", "would", "should", "will", "shall",
    "not", "but", "and", "for", "with", "from", "about",
    "more", "most", "many", "much", "some", "any", "all", "each",
    "also", "just", "than", "then", "only", "very", "well",
}


def extract_entities(text: str, use_spacy: bool = True) -> Set[str]:
    """Extract entities from text. Tries spaCy first, falls back to regex."""
    if use_spacy:
        return _extract_entities_spacy(text)
    return _extract_entities_simple(text)


#Constructing the graph
class EntityGraph:
    """
    A lightweight entity-chunk graph for Graph RAG.
    
    Nodes: entities (strings)
    Edges: entities that co-occur in the same chunk
    Mapping: entity → set of chunk indices
    """

    def __init__(self):
        self.entity_to_chunks: Dict[str, Set[int]] = defaultdict(set)
        self.chunk_to_entities: Dict[int, Set[str]] = defaultdict(set)
        self.entity_cooccurrence: Dict[str, Set[str]] = defaultdict(set)  # entity → co-occurring entities

    def add_chunk(self, chunk_idx: int, entities: Set[str]):
        """Register a chunk and its entities in the graph."""
        for entity in entities:
            self.entity_to_chunks[entity].add(chunk_idx)
            self.chunk_to_entities[chunk_idx].add(entity)

        # Build co-occurrence edges
        entity_list = list(entities)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                self.entity_cooccurrence[entity_list[i]].add(entity_list[j])
                self.entity_cooccurrence[entity_list[j]].add(entity_list[i])

    def get_seed_chunks(self, query_entities: Set[str]) -> Set[int]:
        """Get chunk indices directly linked to query entities."""
        chunks = set()
        for entity in query_entities:
            chunks.update(self.entity_to_chunks.get(entity, set()))
        return chunks

    def expand_by_neighbors(self, query_entities: Set[str], max_hops: int = 1) -> Set[int]:
        """
        Expand from query entities through the co-occurrence graph.
        Returns chunk indices reachable within max_hops.
        """
        # Start with seed entities
        current_entities = set(query_entities)
        all_entities = set(query_entities)

        for _ in range(max_hops):
            next_entities = set()
            for entity in current_entities:
                neighbors = self.entity_cooccurrence.get(entity, set())
                next_entities.update(neighbors - all_entities)
            all_entities.update(next_entities)
            current_entities = next_entities

        # Collect chunks from all discovered entities
        chunks = set()
        for entity in all_entities:
            chunks.update(self.entity_to_chunks.get(entity, set()))
        return chunks


def _build_entity_graph(index: Index) -> EntityGraph:
    """Build entity graph from all chunks in the index."""
    graph = EntityGraph()
    for idx, chunk_text in enumerate(index.chunks):
        entities = extract_entities(chunk_text, use_spacy=True)
        if entities:
            graph.add_chunk(idx, entities)
    return graph


# Module-level cache for the graph
_cached_graph: EntityGraph = None
_cached_index_id: int = None


def _get_or_build_graph(index: Index) -> EntityGraph:
    """Get cached graph or build a new one."""
    global _cached_graph, _cached_index_id
    
    index_id = id(index)
    if _cached_graph is not None and _cached_index_id == index_id:
        return _cached_graph

    print("Building entity graph for Graph RAG (this may take a moment)...")
    _cached_graph = _build_entity_graph(index)
    _cached_index_id = index_id
    print(f"Entity graph built: {len(_cached_graph.entity_to_chunks)} entities, "
          f"{len(_cached_graph.chunk_to_entities)} chunks with entities")
    return _cached_graph



def run(query: str, index: Index, config: dict) -> Dict:
    """
    Running the entire Graph RAG pipeline.

    

    
    """
    import numpy as np
    
    top_k = config.get("top_k", 5)

    # Step 1: Get or build entity graph
    graph = _get_or_build_graph(index)

    # Step 2: Extract entities from query
    query_entities = extract_entities(query, use_spacy=True)

    # Step 3 & 4: Get seed chunks + expand
    if query_entities:
        # Find seed chunks from direct entity matches
        seed_chunks = graph.get_seed_chunks(query_entities)
        # Expand by 1 hop through co-occurrence
        expanded_chunks = graph.expand_by_neighbors(query_entities, max_hops=1)
    else:
        seed_chunks = set()
        expanded_chunks = set()

    # Step 5: Re-rank by vector similarity
    query_embedding = embed_text(query, index)

    # If we have graph-based candidates, re-rank them
    if expanded_chunks:
        candidates = list(expanded_chunks)[:200]  # Limit to avoid too many comparisons
        
        # Score each candidate by cosine similarity
        scored = []
        q_emb = query_embedding.reshape(1, -1)
        
        import faiss as faiss_lib
        faiss_lib.normalize_L2(q_emb)
        
        for chunk_idx in candidates:
            # Reconstruct the embedding from FAISS
            chunk_emb = np.zeros((1, index.faiss_index.d), dtype=np.float32)
            index.faiss_index.reconstruct(chunk_idx, chunk_emb[0])
            sim = float(np.dot(q_emb[0], chunk_emb[0]))
            scored.append((chunk_idx, sim))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored[:top_k]
    else:
        # Fallback: if no entities found, use standard vector retrieval
        results = retrieve(query, index, top_k=top_k)
        return {
            "answer": generate_answer(query, [t for t, _, _ in results], config),
            "retrieved_chunks": [
                {
                    "text": text,
                    "score": round(score, 4),
                    "source": meta.page_name if meta else "",
                    "url": meta.page_url if meta else "",
                }
                for text, score, meta in results
            ],
            "query_entities": list(query_entities),
            "graph_info": {
                "seed_chunks": 0,
                "expanded_chunks": 0,
                "note": "No entities found in query — used standard vector retrieval."
            },
            "pipeline": "graph_rag",
        }

    # Build results
    selected_chunks = []
    for chunk_idx, sim_score in top_indices:
        text = index.chunks[chunk_idx]
        meta = index.metadata[chunk_idx]
        selected_chunks.append((text, sim_score, meta))

    # Step 6: Generate answer
    context_chunks = [text for text, _, _ in selected_chunks]
    answer = generate_answer(query, context_chunks, config)

    return {
        "answer": answer,
        "retrieved_chunks": [
            {
                "text": text,
                "score": round(score, 4),
                "source": meta.page_name if meta else "",
                "url": meta.page_url if meta else "",
            }
            for text, score, meta in selected_chunks
        ],
        "query_entities": list(query_entities),
        "graph_info": {
            "seed_chunks": len(seed_chunks),
            "expanded_chunks": len(expanded_chunks),
            "total_entities_in_graph": len(graph.entity_to_chunks),
        },
        "pipeline": "graph_rag",
    }
