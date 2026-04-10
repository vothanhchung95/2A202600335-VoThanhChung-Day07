from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._chroma_client = chromadb.Client()
            self._collection = self._chroma_client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None
            self._chroma_client = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        record_id = f"{doc.id}_{self._next_index}"
        self._next_index += 1
        return {
            'id': record_id,
            'embedding': embedding,
            'content': doc.content,
            'metadata': doc.metadata,
            'doc_id': doc.id,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)

        scored_results = []
        for record in records:
            score = _dot(query_embedding, record['embedding'])
            scored_results.append({
                'id': record['id'],
                'content': record['content'],
                'metadata': record['metadata'],
                'score': score,
            })

        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)

            if self._use_chroma and self._collection is not None:
                self._collection.add(
                    ids=[record['id']],
                    documents=[record['content']],
                    embeddings=[record['embedding']],
                    metadatas=[record['metadata']],
                )
            else:
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': 1.0 - (results['distances'][0][i] if results['distances'] else 0.0),
                })
            return formatted_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k)

        # Filter records by metadata
        if self._use_chroma and self._collection is not None:
            where_clause = {}
            for key, value in metadata_filter.items():
                where_clause[key] = value
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
            )
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': 1.0 - (results['distances'][0][i] if results['distances'] else 0.0),
                })
            return formatted_results
        else:
            filtered_records = []
            for record in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if record['metadata'].get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            results = self._collection.get(where={'doc_id': doc_id})
            if not results['ids']:
                return False
            self._collection.delete(ids=results['ids'])
            return True
        else:
            original_len = len(self._store)
            self._store = [r for r in self._store if r.get('doc_id') != doc_id]
            return len(self._store) < original_len
