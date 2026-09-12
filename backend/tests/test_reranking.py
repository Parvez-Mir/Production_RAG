import pytest

from app.services.reranking import RerankingError, RerankingManager
from app.services.retrieval import RetrievedChunk


class FakeCrossEncoder:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.scores[f"{query}|{text}"] for query, text in pairs]


def make_chunk(chunk_id: str, text: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        similarity_score=score,
        metadata={"chunk_id": chunk_id},
        source_document="notes.txt",
    )


def test_rerank_reorders_chunks_by_new_relevance_scores() -> None:
    model = FakeCrossEncoder(
        {
            "query|first": 0.2,
            "query|second": 0.8,
            "query|third": 0.6,
        }
    )
    manager = RerankingManager(model=model)
    chunks = [
        make_chunk("first", "first"),
        make_chunk("second", "second"),
        make_chunk("third", "third"),
    ]

    results = manager.rerank("query", chunks, top_k=2)

    assert [chunk.chunk_id for chunk in results] == ["second", "third"]
    assert results[0].similarity_score > results[1].similarity_score


def test_rerank_with_threshold_filters_weak_matches() -> None:
    model = FakeCrossEncoder(
        {
            "query|weak": 0.2,
            "query|strong": 0.9,
        }
    )
    manager = RerankingManager(model=model)
    chunks = [make_chunk("weak", "weak"), make_chunk("strong", "strong")]

    results = manager.rerank_with_threshold("query", chunks, top_k=2, threshold=0.5)

    assert [chunk.chunk_id for chunk in results] == ["strong"]


@pytest.mark.parametrize(
    ("query", "top_k", "threshold"),
    [("", 3, 0.5), ("query", 0, 0.5), ("query", 3, 1.5)],
)
def test_invalid_inputs_are_rejected(query: str, top_k: int, threshold: float) -> None:
    manager = RerankingManager(model=FakeCrossEncoder({}))

    with pytest.raises(RerankingError):
        manager.rerank_with_threshold(query, [make_chunk("x", "text")], top_k=top_k, threshold=threshold)
