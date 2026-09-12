import pytest

from app.services.bm25 import BM25SearchError, BM25SearchManager
from app.services.retrieval import RetrievedChunk


def make_chunk(chunk_id: str, text: str, document_id: str = "doc-1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        similarity_score=0.0,
        metadata={"chunk_id": chunk_id, "doc_id": document_id},
        source_document=f"{document_id}.txt",
    )


def test_search_prioritizes_exact_keyword_matches() -> None:
    manager = BM25SearchManager(
        [
            make_chunk("semantic", "The service has a network failure."),
            make_chunk("exact", "ERR_CONNECTION_RESET occurs when the connection resets."),
            make_chunk("other", "The service returns a response."),
        ]
    )

    results = manager.search("ERR_CONNECTION_RESET")

    assert [chunk.chunk_id for chunk in results] == ["exact"]
    assert results[0].similarity_score == 1.0


def test_search_normalizes_case_and_punctuation_and_limits_results() -> None:
    manager = BM25SearchManager(
        [
            make_chunk("one", "Python, testing, and fixtures."),
            make_chunk("two", "Testing Python applications."),
        ]
    )

    results = manager.search("FIXTURES!", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "one"
    assert 0.0 <= results[0].similarity_score <= 1.0


def test_add_and_remove_document_update_the_index() -> None:
    manager = BM25SearchManager([make_chunk("one", "initial content")])
    manager.add_chunks([make_chunk("two", "new keyword", document_id="doc-2")])

    assert [chunk.chunk_id for chunk in manager.search("keyword")] == ["two"]

    manager.remove_document("doc-2")

    assert manager.search("keyword") == []


def test_empty_and_unmatched_queries_are_handled() -> None:
    manager = BM25SearchManager([make_chunk("one", "indexed content")])

    with pytest.raises(BM25SearchError, match="empty"):
        manager.search("!!!")
    assert manager.search("missing") == []


def test_duplicate_chunk_ids_are_replaced() -> None:
    manager = BM25SearchManager([make_chunk("one", "old content")])
    manager.add_chunks([make_chunk("one", "new keyword")])

    assert [chunk.chunk_id for chunk in manager.search("keyword")] == ["one"]
    assert manager.search("old") == []
