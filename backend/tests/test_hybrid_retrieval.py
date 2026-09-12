import pytest

from app.services.hybrid_retrieval import HybridRetrievalError, HybridRetriever
from app.services.retrieval import RetrievedChunk


def make_chunk(chunk_id: str, score: float, text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text or chunk_id,
        similarity_score=score,
        metadata={"chunk_id": chunk_id},
        source_document="notes.txt",
    )


class FakeVectorRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results


class FakeKeywordRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results


def test_retrieve_runs_both_searches_deduplicates_and_combines_scores() -> None:
    vector = FakeVectorRetriever(
        [make_chunk("shared", 0.8), make_chunk("semantic", 0.7)]
    )
    keyword = FakeKeywordRetriever(
        [make_chunk("shared", 0.4), make_chunk("keyword", 1.0)]
    )

    results = HybridRetriever(vector, keyword).retrieve("query", top_k=3)

    assert [result.chunk_id for result in results] == ["shared", "semantic", "keyword"]
    assert results[0].similarity_score == pytest.approx(0.68)
    assert vector.calls == [("query", 10)]
    assert keyword.calls == [("query", 10)]


def test_custom_weights_change_ranking() -> None:
    vector = FakeVectorRetriever([make_chunk("semantic", 1.0), make_chunk("keyword", 0.0)])
    keyword = FakeKeywordRetriever([make_chunk("keyword", 1.0)])

    results = HybridRetriever(vector, keyword).retrieve_weighted(
        "query",
        top_k=2,
        vector_weight=0.2,
        bm25_weight=0.8,
    )

    assert [result.chunk_id for result in results] == ["keyword", "semantic"]


@pytest.mark.parametrize(
    ("vector_weight", "bm25_weight"),
    [(-0.1, 1.0), (0.0, 0.0)],
)
def test_invalid_weights_are_rejected(vector_weight: float, bm25_weight: float) -> None:
    vector = FakeVectorRetriever([])
    keyword = FakeKeywordRetriever([])

    with pytest.raises(HybridRetrievalError):
        HybridRetriever(vector, keyword, vector_weight, bm25_weight)


def test_invalid_top_k_is_rejected() -> None:
    manager = HybridRetriever(FakeVectorRetriever([]), FakeKeywordRetriever([]))

    with pytest.raises(HybridRetrievalError, match="top_k"):
        manager.retrieve("query", top_k=0)
