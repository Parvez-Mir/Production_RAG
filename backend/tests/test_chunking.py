from app.config import Settings
from app.services.chunking import ChunkingFactory, SemanticChunkingStrategy
from app.services.parsers import Document


def make_document(content: str) -> Document:
    return Document(content=content, metadata={"source": "notes.txt", "file_type": "txt"})


def test_semantic_chunking_respects_token_limit_and_metadata() -> None:
    settings = Settings(chunk_size_tokens=20, chunk_overlap_tokens=5)
    document = make_document(
        "First sentence has useful context. Second sentence adds more detail. "
        "Third sentence continues the explanation. Fourth sentence closes it."
    )

    chunks = SemanticChunkingStrategy(settings).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.metadata["token_count"] <= 20 for chunk in chunks)
    assert all(chunk.metadata["source"] == "notes.txt" for chunk in chunks)
    assert [chunk.metadata["position"] for chunk in chunks] == list(range(len(chunks)))


def test_chunking_preserves_sentence_boundaries_and_overlap() -> None:
    settings = Settings(chunk_size_tokens=12, chunk_overlap_tokens=5)
    document = make_document(
        "Alpha starts here. Beta carries the shared context. "
        "Gamma adds the next idea. Delta finishes the document."
    )

    chunks = ChunkingFactory.chunk(document, settings=settings)

    assert all(not chunk.text.endswith(("Alpha", "Beta", "Gamma")) for chunk in chunks)
    assert any(set(first.text.split()) & set(second.text.split()) for first, second in zip(chunks, chunks[1:]))


def test_empty_documents_produce_no_chunks() -> None:
    assert ChunkingFactory.chunk(make_document("\n\n")) == []


def test_long_sentence_is_split_without_exceeding_limit() -> None:
    settings = Settings(chunk_size_tokens=10, chunk_overlap_tokens=2)
    content = " ".join(f"word{index}" for index in range(40)) + "."

    chunks = ChunkingFactory.chunk(make_document(content), settings=settings)

    assert len(chunks) > 1
    assert all(chunk.metadata["token_count"] <= 10 for chunk in chunks)