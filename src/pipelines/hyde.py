"""
HyDE: generate hypothetical document, retrieve from global index by similarity to it, generate final answer.
"""

import numpy as np
from typing import Dict, List

from src.retrieval import retrieve_by_embedding, embed_text
from src.generation import generate_answer, generate_text
from src.corpus import Index


def _generate_hypothetical_document(query: str, config: dict) -> str:
    """
    We use the LLM to generate a hypothetical passage that might contain the answer.
    """
    prompt = (
        "Given the following question, write a short passage (1-2 paragraphs) that would "
        "contain the answer to this question. Write it as if it were an excerpt from a "
        "factual article or webpage that directly answers the question. "
        "Do not say 'I don't know' or hedge — just write the passage as if you know the answer.\n\n"
        f"Question: {query}\n\n"
        "Hypothetical passage:"
    )

    return generate_text(
        prompt, config,
        system_prompt="You are a factual content writer. Write passages that directly answer questions."
    )


def run(query: str, index: Index, config: dict) -> Dict:
    """
    Run HyDE (Hypothetical Document Embedding) pipeline.

    
    """
    top_k = config.get("top_k", 5)

    # Step 1: Generate hypothetical document
    hypothetical_doc = _generate_hypothetical_document(query, config)

    # Step 2: Embed the hypothetical document
    hyp_embedding = embed_text(hypothetical_doc, index)

    # Step 3: Retrieve from global index using hypothetical embedding
    results = retrieve_by_embedding(hyp_embedding, index, top_k=top_k)

    # Step 4: Generate FINAL answer from retrieved chunks (not the hypothetical doc)
    context_chunks = [text for text, _, _ in results]
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
            for text, score, meta in results
        ],
        "hypothetical_document": hypothetical_doc,
        "pipeline": "hyde",
    }
