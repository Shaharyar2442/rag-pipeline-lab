"""
LLM answer generation from (query, retrieved context).
Supports Groq and Google Gemini as providers.
Do not change this module's location in the project.
"""

from typing import List, Optional


def _call_groq(prompt: str, config: dict, system_prompt: str = None) -> str:
    """Call Groq API with the given prompt."""
    from groq import Groq

    client = Groq(api_key=config["api_key"])

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.get("generation_model", "llama-3.3-70b-versatile"),
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str, config: dict, system_prompt: str = None) -> str:
    """Call Google Gemini API with the given prompt."""
    import google.generativeai as genai

    genai.configure(api_key=config["api_key"])
    model = genai.GenerativeModel(
        config.get("generation_model", "gemini-1.5-flash"),
        system_instruction=system_prompt,
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_text(prompt: str, config: dict, system_prompt: str = None) -> str:
    """
    Generate text from a prompt using the configured LLM provider.
    This is the low-level generation function used by pipelines for
    sub-tasks like generating query variants, hypothetical docs, etc.

    Args:
        prompt: The user/task prompt.
        config: Config dict with api_provider, api_key, generation_model.
        system_prompt: Optional system-level instruction.

    Returns:
        Generated text string.
    """
    provider = config.get("api_provider", "groq").lower()

    if provider == "groq":
        return _call_groq(prompt, config, system_prompt)
    elif provider == "gemini":
        return _call_gemini(prompt, config, system_prompt)
    else:
        raise ValueError(f"Unsupported API provider: {provider}. Use 'groq' or 'gemini'.")


def generate_answer(
    query: str,
    context_chunks: List[str],
    config: dict,
    system_prompt: str = None,
) -> str:
    """
    Generate an answer to a query using retrieved context chunks.

    Args:
        query: The user's question.
        context_chunks: List of retrieved text chunks to use as context.
        config: Config dict with LLM settings.
        system_prompt: Optional custom system prompt. If None, uses a default RAG prompt.

    Returns:
        Generated answer string.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a factual question-answering assistant. "
            "Answer questions as concisely as possible — ideally with just the answer itself (a name, number, date, etc.). "
            "Use the provided context AND your knowledge to answer. "
            "Do NOT say 'the context does not contain' or 'I cannot determine'. "
            "Always give your best factual answer. Keep it SHORT."
        )

    # Build context string
    context_str = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_str += f"[Source {i}]: {chunk}\n\n"

    prompt = (
        f"Context:\n{context_str}\n"
        f"Question: {query}\n\n"
        f"Give a short, direct answer. Just the answer — no explanation needed."
    )

    return generate_text(prompt, config, system_prompt)


def generate_answer_with_citations(
    query: str,
    context_chunks: List[str],
    source_names: List[str],
    source_urls: List[str],
    config: dict,
) -> str:
    """
    Generate an answer with IEEE-style citations.
    Used by CRAG pipeline.

    Args:
        query: The user's question.
        context_chunks: Retrieved text chunks.
        source_names: Page names for each chunk.
        source_urls: Page URLs for each chunk.
        config: Config dict.

    Returns:
        Answer string with citations appended.
    """
    system_prompt = (
        "You are a factual question-answering assistant. "
        "Answer concisely — give the direct answer first, then cite sources using [1], [2], etc. "
        "Use the provided context AND your knowledge. Do NOT refuse to answer."
    )

    context_str = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_str += f"[{i}] {chunk}\n\n"

    prompt = (
        f"Context:\n{context_str}\n"
        f"Question: {query}\n\n"
        f"Give a short, direct answer and cite sources using [1], [2], etc."
    )

    answer = generate_text(prompt, config, system_prompt)

    # Append formal citation list
    citation_block = "\n\nReferences:\n"
    for i, (name, url) in enumerate(zip(source_names, source_urls), 1):
        citation_block += f"[{i}] {name}, {url}\n"

    return answer + citation_block
