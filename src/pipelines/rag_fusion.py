"""
RAG Fusion: multiple queries, retrieve from global index for each, merge ranked lists (e.g. RRF), generate answer.
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict

from src.retrieval import retrieve, embed_text
from src.generation import generate_answer, generate_text
from src.corpus import Index


def _generate_query_variants(query: str, config: dict, num_variants: int = 4) -> List[str]:
    """
    Use LLM to generate multiple query variants for the original query.
    """
    prompt = (
        f"Generate {num_variants} different search queries that could help answer this question. "
        f"Each query should approach the question from a different angle or use different keywords. "
        f"Return ONLY the queries, one per line, numbered 1-{num_variants}. No explanations.\n\n"
        f"Original question: {query}"
    )

    response = generate_text(prompt, config, system_prompt="You are a search query generator. Generate diverse search queries.")

    # Parse the response into individual queries
    variants = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1.", "1)", "1:", etc.
        for prefix in [".", ")", ":", "-"]:
            if len(line) > 2 and line[0].isdigit() and prefix in line[:3]:
                line = line.split(prefix, 1)[-1].strip()
                break
        if line and len(line) > 5:
            variants.append(line)

    # We always include the original query
    return [query] + variants[:num_variants]


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float, Any]]],
    k: int = 60,
) -> List[Tuple[str, float, Any]]:
    """
    Merging multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF_score(chunk) = sum(1 / (k + rank_i)) for each list where chunk appears.

    
    """
    chunk_scores = defaultdict(float)
    chunk_data = {}  # chunk_text -> (score, metadata)

    for ranked_list in ranked_lists:
        for rank, (text, score, meta) in enumerate(ranked_list, 1):
            chunk_scores[text] += 1.0 / (k + rank)
            # Keep the best original score and metadata
            if text not in chunk_data or score > chunk_data[text][0]:
                chunk_data[text] = (score, meta)

    # Sort by fused score
    fused = []
    for text, fused_score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
        original_score, meta = chunk_data[text]
        fused.append((text, fused_score, meta))

    return fused


def run(query: str, index: Index, config: dict) -> Dict:
    """
    Running the entire RAG Fusion pipeline.
    """
    top_k = config.get("top_k", 5)
    retrieve_per_query = top_k * 2  # Retrieve more per query for better fusion

    # Step 1: Generate query variants
    query_variants = _generate_query_variants(query, config)

    # Step 2: Retrieve for each variant
    ranked_lists = []
    for variant in query_variants:
        results = retrieve(variant, index, top_k=retrieve_per_query)
        ranked_lists.append(results)

    # Step 3: Fuse with RRF
    fused = _reciprocal_rank_fusion(ranked_lists)

    # Step 4: Take top-k
    top_results = fused[:top_k]

    # Step 5: Generate answer
    context_chunks = [text for text, _, _ in top_results]
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
            for text, score, meta in top_results
        ],
        "query_variants": query_variants,
        "pipeline": "rag_fusion",
    }
