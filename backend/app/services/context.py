from dataclasses import dataclass
from typing import Any

from app.services.retrieval import RetrievedChunk


@dataclass(frozen=True)
class ChunkSource:
    chunk_id: str
    document: str
    section: str | None
    excerpt: str


@dataclass(frozen=True)
class ContextOutput:
    context_text: str
    chunk_sources: list[ChunkSource]


class ContextBuilder:
    """Format ranked chunks into LLM context while retaining citation data."""

    HEADER = "Context from documents:"
    SEPARATOR = "\n\n"

    def build(self, chunks: list[RetrievedChunk]) -> ContextOutput:
        formatted_chunks: list[str] = []
        chunk_sources: list[ChunkSource] = []

        for chunk in chunks:
            document = self._document_name(chunk)
            section = self._section_name(chunk)
            location = f"[Document: {document}"
            if section:
                location += f", Section {section}"
            location += "]"

            formatted_chunks.append(f"{location}\n{chunk.text}")
            chunk_sources.append(
                ChunkSource(
                    chunk_id=chunk.chunk_id,
                    document=document,
                    section=section,
                    excerpt=chunk.text,
                )
            )

        if not formatted_chunks:
            return ContextOutput(context_text=self.HEADER, chunk_sources=[])

        return ContextOutput(
            context_text=self.HEADER + self.SEPARATOR + self.SEPARATOR.join(formatted_chunks),
            chunk_sources=chunk_sources,
        )

    def _document_name(self, chunk: RetrievedChunk) -> str:
        return str(chunk.metadata.get("source") or chunk.source_document or "unknown")

    def _section_name(self, chunk: RetrievedChunk) -> str | None:
        section: Any = chunk.metadata.get("section")
        if section is None or not str(section).strip():
            return None
        return str(section)