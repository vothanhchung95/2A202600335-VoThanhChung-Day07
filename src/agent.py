from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Retrieve top_k relevant chunks
        results = self._store.search(question, top_k=top_k)

        # Build prompt with context
        context_chunks = [r['content'] for r in results]
        context_text = '\n\n'.join(context_chunks)

        prompt = f"""Context:
        {context_text}

        Question: {question}
        Answer:"""

        # Call LLM and return answer
        return self._llm_fn(prompt)
