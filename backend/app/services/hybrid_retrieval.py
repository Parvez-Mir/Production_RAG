import logging
import time
from typing import Protocol

from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRetrievalError(Exception):
    """Raised when hybrid retrieval arguments are invalid."""


class VectorRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return vector-search results."""


class KeywordRetriever(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return keyword-search results."""


class HybridRetriever:
    CANDIDATE_TOP_K = 10

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self._validate_weights(vector_weight, bm25_weight)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.retrieve_weighted(
            query,
            top_k=top_k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
        )

    def retrieve_weighted(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> list[RetrievedChunk]:
        self._validate_top_k(top_k)
        self._validate_weights(vector_weight, bm25_weight)

        started_at = time.perf_counter()
        vector_results = self.vector_retriever.retrieve(query, self.CANDIDATE_TOP_K)
        keyword_results = self.keyword_retriever.search(query, self.CANDIDATE_TOP_K)
        merged = self._merge_results(
            vector_results,
            keyword_results,
            vector_weight,
            bm25_weight,
        )
        logger.info(
            "Hybrid retrieval returned %d chunks in %.3fs",
            len(merged[:top_k]),
            time.perf_counter() - started_at,
        )
        return merged[:top_k]

    def _merge_results(
        self,
        vector_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
        vector_weight: float,
        bm25_weight: float,
    ) -> list[RetrievedChunk]:
        merged: dict[str, dict[str, object]] = {}
        for result in vector_results:
            merged[result.chunk_id] = {
                "chunk": result,
                "vector_score": result.similarity_score,
                "bm25_score": 0.0,
            }
        for result in keyword_results:
            entry = merged.setdefault(
                result.chunk_id,
                {"chunk": result, "vector_score": 0.0, "bm25_score": 0.0},
            )
            entry["bm25_score"] = result.similarity_score

        results: list[RetrievedChunk] = []
        total_weight = vector_weight + bm25_weight
        for entry in merged.values():
            chunk = entry["chunk"]
            assert isinstance(chunk, RetrievedChunk)
            combined_score = (
                vector_weight * float(entry["vector_score"])
                + bm25_weight * float(entry["bm25_score"])
            ) / total_weight
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=combined_score,
                    metadata=chunk.metadata,
                    source_document=chunk.source_document,
                )
            )
        return sorted(
            results,
            key=lambda result: (-result.similarity_score, result.chunk_id),
        )

    def _validate_weights(self, vector_weight: float, bm25_weight: float) -> None:
        if vector_weight < 0 or bm25_weight < 0:
            raise HybridRetrievalError("Retrieval weights must be non-negative")
        if vector_weight + bm25_weight == 0:
            raise HybridRetrievalError("At least one retrieval weight must be positive")

    def _validate_top_k(self, top_k: int) -> None:
        if top_k < 1:
            raise HybridRetrievalError("top_k must be at least 1")
