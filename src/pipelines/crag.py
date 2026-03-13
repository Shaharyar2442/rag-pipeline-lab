"""
CRAG (Corrective RAG): assess retrieval confidence; use or correct retrieval based on it, then generate.
Includes citations in IEEE style.
"""

from typing import Dict, List, Tuple

from src.retrieval import retrieve
from src.generation import generate_answer_with_citations, generate_answer, generate_text
from src.corpus import Index


# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.65
LOW_CONFIDENCE_THRESHOLD = 0.35


def _assess_confidence_llm(query: str, chunks: List[Tuple[str, float, object]], config: dict) -> List[float]:
    """
    We assess retrieval confidence using LLM as a judge and for each chunk we ask LLM if chunk is relevant to query returning confidence score
    
    """
    scores = []
    for text, sim_score, meta in chunks:
        prompt = (
            f"On a scale of 0 to 10, how relevant is the following passage to answering the question?\n\n"
            f"Question: {query}\n\n"
            f"Passage: {text[:500]}\n\n"
            f"Reply with ONLY a number from 0 to 10. Nothing else."
        )
        try:
            response = generate_text(prompt, config, system_prompt="You are a relevance judge. Reply with only a number.")
            # Extract the number
            num_str = ''.join(c for c in response.strip() if c.isdigit() or c == '.')
            score = float(num_str) / 10.0 if num_str else sim_score
            score = max(0.0, min(1.0, score))
        except Exception:
            # Fall back to cosine similarity score
            score = max(0.0, min(1.0, sim_score))
        scores.append(score)

    return scores


def _assess_confidence_simple(query: str, chunks: List[Tuple[str, float, object]]) -> List[float]:
    """
    Simple confidence assessment based on cosine similarity scores.
    """
    return [max(0.0, min(1.0, score)) for _, score, _ in chunks]


def run(query: str, index: Index, config: dict) -> Dict:
    """
    Running CRAG (Corrective RAG) pipeline.
    """
    top_k = config.get("top_k", 5)

    # Step 1: Standard retrieval
    results = retrieve(query, index, top_k=top_k)

    if not results:
        answer = generate_text(
            f"Answer this question concisely: {query}",
            config,
            system_prompt="You are a helpful assistant. Answer concisely."
        )
        return {
            "answer": answer,
            "retrieved_chunks": [],
            "confidence_level": "none",
            "confidence_scores": [],
            "pipeline": "crag",
        }

    # Step 2: Assess confidence
    # Use simple similarity-based scoring (fast) for evaluation
    # Use LLM judge for interactive queries if desired
    use_llm_judge = config.get("crag_use_llm_judge", False)

    if use_llm_judge:
        confidence_scores = _assess_confidence_llm(query, results, config)
    else:
        confidence_scores = _assess_confidence_simple(query, results)

    mean_confidence = sum(confidence_scores) / len(confidence_scores)

    # Step 3: Decision gate
    if mean_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "high"
        # Use all retrieved chunks
        selected = results
        selected_scores = confidence_scores
    elif mean_confidence >= LOW_CONFIDENCE_THRESHOLD:
        confidence_level = "medium"
        # Filter to chunks above the low threshold
        selected = []
        selected_scores = []
        for (text, score, meta), conf in zip(results, confidence_scores):
            if conf >= LOW_CONFIDENCE_THRESHOLD:
                selected.append((text, score, meta))
                selected_scores.append(conf)
        if not selected:
            # If all filtered out, keep top 1
            selected = [results[0]]
            selected_scores = [confidence_scores[0]]
    else:
        confidence_level = "low"
        # Fall back to LLM-only generation (no retrieval context)
        answer = generate_text(
            f"Answer this question with just the answer — a name, number, or short phrase: {query}",
            config,
            system_prompt="You are a factual QA assistant. Give only the direct answer, nothing else."
        )
        return {
            "answer": answer,
            "retrieved_chunks": [
                {
                    "text": text,
                    "score": round(score, 4),
                    "confidence": round(conf, 4),
                    "source": meta.page_name if meta else "",
                    "url": meta.page_url if meta else "",
                }
                for (text, score, meta), conf in zip(results, confidence_scores)
            ],
            "confidence_level": "low",
            "confidence_scores": [round(c, 4) for c in confidence_scores],
            "mean_confidence": round(mean_confidence, 4),
            "note": "Low confidence in retrieval — answer generated without retrieval context.",
            "pipeline": "crag",
        }

    # Step 4: Generate with citations
    context_chunks = [text for text, _, _ in selected]
    source_names = [meta.page_name if meta else f"Source {i+1}" for i, (_, _, meta) in enumerate(selected)]
    source_urls = [meta.page_url if meta else "" for _, _, meta in selected]

    answer = generate_answer_with_citations(
        query, context_chunks, source_names, source_urls, config
    )

    return {
        "answer": answer,
        "retrieved_chunks": [
            {
                "text": text,
                "score": round(score, 4),
                "confidence": round(conf, 4),
                "source": meta.page_name if meta else "",
                "url": meta.page_url if meta else "",
            }
            for (text, score, meta), conf in zip(selected, selected_scores)
        ],
        "confidence_level": confidence_level,
        "confidence_scores": [round(c, 4) for c in confidence_scores],
        "mean_confidence": round(mean_confidence, 4),
        "pipeline": "crag",
    }
