from app.services.context import ContextBuilder
from app.services.retrieval import RetrievedChunk


def make_chunk(
    chunk_id: str,
    text: str,
    source_document: str = "fallback.txt",
    metadata: dict[str, object] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        similarity_score=0.9,
        metadata=metadata or {},
        source_document=source_document,
    )


def test_build_formats_chunks_in_ranked_order_and_tracks_sources() -> None:
    chunks = [
        make_chunk(
            "chunk-1",
            "First text.",
            metadata={"source": "report.pdf", "section": "2.1"},
        ),
        make_chunk("chunk-2", "Second text.", metadata={"source": "notes.md"}),
    ]

    result = ContextBuilder().build(chunks)

    assert result.context_text == (
        "Context from documents:\n\n"
        "[Document: report.pdf, Section 2.1]\nFirst text.\n\n"
        "[Document: notes.md]\nSecond text."
    )
    assert result.chunk_sources[0].chunk_id == "chunk-1"
    assert result.chunk_sources[0].document == "report.pdf"
    assert result.chunk_sources[0].section == "2.1"
    assert result.chunk_sources[0].excerpt == "First text."


def test_build_handles_missing_metadata_and_empty_chunks() -> None:
    builder = ContextBuilder()

    missing_metadata = builder.build([make_chunk("chunk-1", "Text.")])
    empty = builder.build([])

    assert "[Document: fallback.txt]" in missing_metadata.context_text
    assert missing_metadata.chunk_sources[0].section is None
    assert empty.context_text == "Context from documents:"
    assert empty.chunk_sources == []