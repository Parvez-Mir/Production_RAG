from types import SimpleNamespace

import pytest

from app.services.retrieval import RetrievedChunk, RetrievalError, RetrieverManager


class FakeEmbeddings:
    def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
        **kwargs: object,
    ) -> list[tuple[SimpleNamespace, float]]:
        self.calls.append({"query": query, "k": k, **kwargs})
        return [
            (
                SimpleNamespace(
                    page_content="lower relevance",
                    metadata={"chunk_id": "chunk-1", "source": "notes.txt"},
                ),
                0.45,
            ),
            (
                SimpleNamespace(
                    page_content="highest relevance",
                    metadata={"chunk_id": "chunk-2", "source": "report.pdf"},
                ),
                0.9,
            ),
        ]


def make_manager(vector_store: FakeVectorStore) -> RetrieverManager:
    return RetrieverManager(
        vector_db=SimpleNamespace(),
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )


def test_retrieve_returns_ranked_chunks_and_respects_top_k() -> None:
    vector_store = FakeVectorStore()
    manager = make_manager(vector_store)

    results = manager.retrieve("machine learning", top_k=2, threshold=0.3)

    assert results == [
        RetrievedChunk(
            chunk_id="chunk-2",
            text="highest relevance",
            similarity_score=0.9,
            metadata={"chunk_id": "chunk-2", "source": "report.pdf"},
            source_document="report.pdf",
        ),
        RetrievedChunk(
            chunk_id="chunk-1",
            text="lower relevance",
            similarity_score=0.45,
            metadata={"chunk_id": "chunk-1", "source": "notes.txt"},
            source_document="notes.txt",
        ),
    ]
    assert vector_store.calls == [{"query": "machine learning", "k": 2}]


def test_retrieve_applies_threshold() -> None:
    vector_store = FakeVectorStore()
    manager = make_manager(vector_store)

    results = manager.retrieve("query", threshold=0.5)

    assert [result.chunk_id for result in results] == ["chunk-2"]


def test_retrieve_with_filter_forwards_filter() -> None:
    vector_store = FakeVectorStore()
    manager = make_manager(vector_store)
    metadata_filter = object()

    manager.retrieve_with_filter("query", metadata_filter=metadata_filter)

    assert vector_store.calls[0]["filters"] is metadata_filter


@pytest.mark.parametrize(
    ("query", "message"),
    [("", "Query cannot be empty"), ("   ", "Query cannot be empty")],
)
def test_retrieve_rejects_empty_queries(query: str, message: str) -> None:
    with pytest.raises(RetrievalError, match=message):
        make_manager(FakeVectorStore()).retrieve(query)


def test_retrieve_rejects_invalid_top_k_and_threshold() -> None:
    manager = make_manager(FakeVectorStore())

    with pytest.raises(RetrievalError, match="top_k"):
        manager.retrieve("query", top_k=0)
    with pytest.raises(RetrievalError, match="threshold"):
        manager.retrieve("query", threshold=1.1)