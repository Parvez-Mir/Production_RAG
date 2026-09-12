import math
import re
from collections import Counter, defaultdict

from app.services.retrieval import RetrievedChunk


class BM25SearchError(Exception):
    """Raised when BM25 search arguments are invalid."""


class BM25SearchManager:
    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise BM25SearchError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise BM25SearchError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self._chunks: dict[str, RetrievedChunk] = {}
        self._tokenized_chunks: dict[str, list[str]] = {}
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._inverted_index: dict[str, set[str]] = defaultdict(set)
        self._average_document_length = 0.0
        self.add_chunks(chunks or [])

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_tokens = self._tokenize(query)
        self._validate_query(query_tokens)
        self._validate_top_k(top_k)

        candidate_ids = {
            chunk_id
            for token in set(query_tokens)
            for chunk_id in self._inverted_index.get(token, set())
        }
        if not candidate_ids:
            return []

        query_term_counts = Counter(query_tokens)
        scores = {
            chunk_id: self._score(chunk_id, query_term_counts)
            for chunk_id in candidate_ids
        }
        maximum_score = max(scores.values(), default=0.0)
        ranked_ids = sorted(
            candidate_ids,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )[:top_k]

        return [
            self._with_score(
                self._chunks[chunk_id],
                scores[chunk_id] / maximum_score if maximum_score else 0.0,
            )
            for chunk_id in ranked_ids
        ]

    def add_chunks(self, chunks: list[RetrievedChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._rebuild_index()

    def remove_document(self, document_id: str) -> None:
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.metadata.get("doc_id") != document_id
            and chunk.source_document != document_id
        }
        self._rebuild_index()

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())

    def _rebuild_index(self) -> None:
        self._tokenized_chunks = {
            chunk_id: self._tokenize(chunk.text)
            for chunk_id, chunk in self._chunks.items()
        }
        self._term_frequencies = {
            chunk_id: Counter(tokens)
            for chunk_id, tokens in self._tokenized_chunks.items()
        }
        self._document_frequency = Counter()
        self._inverted_index = defaultdict(set)
        for chunk_id, tokens in self._tokenized_chunks.items():
            for token in set(tokens):
                self._document_frequency[token] += 1
                self._inverted_index[token].add(chunk_id)

        total_length = sum(len(tokens) for tokens in self._tokenized_chunks.values())
        self._average_document_length = (
            total_length / len(self._tokenized_chunks) if self._tokenized_chunks else 0.0
        )

    def _score(self, chunk_id: str, query_terms: Counter[str]) -> float:
        frequencies = self._term_frequencies[chunk_id]
        document_length = len(self._tokenized_chunks[chunk_id])
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = self._document_frequency[term]
            document_count = len(self._chunks)
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_ratio = (
                document_length / self._average_document_length
                if self._average_document_length
                else 0.0
            )
            denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
            score += (
                inverse_document_frequency
                * (term_frequency * (self.k1 + 1.0) / denominator)
                * query_frequency
            )
        return score

    def _with_score(self, chunk: RetrievedChunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            similarity_score=score,
            metadata=chunk.metadata,
            source_document=chunk.source_document,
        )

    def _validate_query(self, query_tokens: list[str]) -> None:
        if not query_tokens:
            raise BM25SearchError("Query cannot be empty")

    def _validate_top_k(self, top_k: int) -> None:
        if top_k < 1:
            raise BM25SearchError("top_k must be at least 1")
