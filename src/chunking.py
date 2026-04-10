from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        # Split on sentence boundaries: ". ", "! ", "? ", ".\n"
        sentences = re.split(r'(?<=[.!?])\s+|\.\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []

        if not remaining_separators:
            # Base case: split into individual characters
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            # Empty separator means split by character
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        parts = current_text.split(separator)
        chunks = []
        current_chunk_parts = []
        current_length = 0

        for part in parts:
            part_len = len(part)
            separator_len = len(separator) if current_chunk_parts else 0

            if current_length + part_len + separator_len <= self.chunk_size:
                if current_chunk_parts:
                    current_chunk_parts.append(separator + part)
                    current_length += separator_len + part_len
                else:
                    current_chunk_parts.append(part)
                    current_length += part_len
            else:
                if current_chunk_parts:
                    chunks.append(''.join(current_chunk_parts))
                current_chunk_parts = [part]
                current_length = part_len

                if current_length > self.chunk_size:
                    # Need to recurse on this piece
                    sub_chunks = self._split(part, next_separators)
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current_chunk_parts = [sub_chunks[-1]]
                        current_length = len(sub_chunks[-1])

        if current_chunk_parts:
            final_chunk = ''.join(current_chunk_parts)
            if len(final_chunk) > self.chunk_size and next_separators:
                chunks.extend(self._split(final_chunk, next_separators))
            else:
                chunks.append(final_chunk)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    magnitude_a = math.sqrt(sum(x * x for x in vec_a))
    magnitude_b = math.sqrt(sum(x * x for x in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        def avg_length(chunks: list[str]) -> float:
            if not chunks:
                return 0.0
            return sum(len(c) for c in chunks) / len(chunks)

        return {
            'fixed_size': {
                'count': len(fixed_chunks),
                'avg_length': avg_length(fixed_chunks),
                'chunks': fixed_chunks,
            },
            'by_sentences': {
                'count': len(sentence_chunks),
                'avg_length': avg_length(sentence_chunks),
                'chunks': sentence_chunks,
            },
            'recursive': {
                'count': len(recursive_chunks),
                'avg_length': avg_length(recursive_chunks),
                'chunks': recursive_chunks,
            },
        }
