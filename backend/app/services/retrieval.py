from dataclasses import dataclass
from typing import Any

from langchain_weaviate import WeaviateVectorStore

from app.services.embeddings import EmbeddingManager
from app.services.vector_db import VectorDBManager


class RetrievalError(Exception):
    """Raised when vector retrieval cannot be completed."""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    similarity_score: float
    metadata: dict[str, Any]
    source_document: str


class RetrieverManager:
    MAX_TOP_K = 50

    def __init__(
        self,
        vector_db: VectorDBManager,
        embeddings: EmbeddingManager,
        vector_store: Any | None = None,
    ) -> None:
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.vector_store = vector_store or self._create_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = 0.3,
    ) -> list[RetrievedChunk]:
        self._validate_query(query)
        self._validate_top_k(top_k)
        self._validate_threshold(threshold)

        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k,
        )
        return self._to_retrieved_chunks(results, threshold)

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Any | None = None,
        threshold: float | None = 0.3,
    ) -> list[RetrievedChunk]:
        self._validate_query(query)
        self._validate_top_k(top_k)
        self._validate_threshold(threshold)

        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k,
            filters=metadata_filter,
        )
        return self._to_retrieved_chunks(results, threshold)

    def _create_vector_store(self) -> WeaviateVectorStore:
        return WeaviateVectorStore(
            client=self.vector_db.client,
            index_name=self.vector_db.settings.weaviate_collection,
            text_key="text",
            embedding=self.embeddings,
        )

    def _to_retrieved_chunks(
        self,
        results: list[tuple[Any, float]],
        threshold: float | None,
    ) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for document, score in results:
            metadata = dict(document.metadata or {})
            chunk_id = str(metadata.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen_ids:
                continue

            similarity_score = max(0.0, min(1.0, float(score)))
            if threshold is not None and similarity_score < threshold:
                continue

            seen_ids.add(chunk_id)
            source_document = str(
                metadata.get("source", metadata.get("doc_id", "unknown"))
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=document.page_content,
                    similarity_score=similarity_score,
                    metadata=metadata,
                    source_document=source_document,
                )
            )

        return sorted(chunks, key=lambda chunk: chunk.similarity_score, reverse=True)

    def _validate_query(self, query: str) -> None:
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")

    def _validate_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= self.MAX_TOP_K:
            raise RetrievalError(f"top_k must be between 1 and {self.MAX_TOP_K}")

    def _validate_threshold(self, threshold: float | None) -> None:
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise RetrievalError("threshold must be between 0 and 1")