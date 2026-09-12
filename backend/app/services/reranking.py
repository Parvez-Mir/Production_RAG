import logging
from collections.abc import Sequence
from typing import Protocol

from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankingError(Exception):
    """Raised when reranking arguments are invalid."""


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return a relevance score for each query/text pair."""


class RerankingManager:
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model: CrossEncoderLike | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.batch_size = batch_size
        self._validate_threshold(threshold)
        self._validate_batch_size(batch_size)

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self._validate_query(query)
        self._validate_top_k(top_k)
        if not chunks:
            return []

        if self.model is None:
            return self._fallback_rerank(query, chunks, top_k)

        ranked = self._compute_scores(query, list(chunks), top_k=None)
        return ranked[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        self._validate_query(query)
        self._validate_top_k(top_k)
        effective_threshold = self.threshold if threshold is None else threshold
        self._validate_threshold(effective_threshold)

        if not chunks:
            return []

        if self.model is None:
            filtered = [chunk for chunk in chunks if chunk.similarity_score >= effective_threshold]
            return self._fallback_rerank(query, filtered, top_k)

        ranked = self._compute_scores(query, list(chunks), top_k=None)
        filtered = [chunk for chunk in ranked if chunk.similarity_score >= effective_threshold]
        return filtered[:top_k]

    def _compute_scores(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None,
    ) -> list[RetrievedChunk]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=True):
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=max(0.0, min(1.0, float(score))),
                    metadata=chunk.metadata,
                    source_document=chunk.source_document,
                )
            )

        reranked.sort(key=lambda chunk: chunk.similarity_score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

    def _fallback_rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        query_tokens = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_tokens = set(chunk.text.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            score = min(1.0, overlap / max(len(query_tokens), 1))
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=score,
                    metadata=chunk.metadata,
                    source_document=chunk.source_document,
                )
            )
        scored.sort(key=lambda chunk: chunk.similarity_score, reverse=True)
        return scored[:top_k]

    def _validate_query(self, query: str) -> None:
        if not query or not query.strip():
            raise RerankingError("Query cannot be empty")

    def _validate_top_k(self, top_k: int) -> None:
        if top_k < 1:
            raise RerankingError("top_k must be at least 1")

    def _validate_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise RerankingError("threshold must be between 0 and 1")

    def _validate_batch_size(self, batch_size: int) -> None:
        if batch_size < 1:
            raise RerankingError("batch_size must be at least 1")
