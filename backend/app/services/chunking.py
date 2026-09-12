import re
from dataclasses import dataclass
from typing import Any, Protocol

import tiktoken

from app.config import Settings, get_settings
from app.services.parsers import Document


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


class ChunkingStrategy(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""


class SemanticChunkingStrategy:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.max_tokens = settings.chunk_size_tokens
        self.overlap_tokens = min(settings.chunk_overlap_tokens, self.max_tokens - 1)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.content.strip():
            return []

        chunks: list[Chunk] = []
        current_units: list[str] = []
        current_tokens = 0
        section = 0

        for paragraph in re.split(r"\n\s*\n", document.content.strip()):
            if not paragraph.strip():
                continue
            sentences = self._split_sentences(paragraph)
            for sentence in sentences:
                for unit in self._split_oversized_sentence(sentence):
                    unit_tokens = self._token_count(unit)
                    separator_tokens = 1 if current_units else 0
                    if current_units and current_tokens + separator_tokens + unit_tokens > self.max_tokens:
                        chunks.append(self._make_chunk(current_units, document, section))
                        current_units, current_tokens = self._overlap(current_units)
                        section += 1

                    current_units.append(unit)
                    current_tokens = self._token_count(" ".join(current_units))

        if current_units:
            chunks.append(self._make_chunk(current_units, document, section))

        return chunks

    def _split_sentences(self, paragraph: str) -> list[str]:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]

    def _split_oversized_sentence(self, sentence: str) -> list[str]:
        if self._token_count(sentence) <= self.max_tokens:
            return [sentence]

        words = sentence.split()
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and self._token_count(candidate) > self.max_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _overlap(self, units: list[str]) -> tuple[list[str], int]:
        overlap: list[str] = []
        for unit in reversed(units):
            candidate = [unit, *overlap]
            if self._token_count(" ".join(candidate)) > self.overlap_tokens:
                break
            overlap = candidate
        return overlap, self._token_count(" ".join(overlap))

    def _make_chunk(self, units: list[str], document: Document, position: int) -> Chunk:
        metadata = {
            **document.metadata,
            "chunk_id": f"{document.metadata.get('source', 'document')}-{position}",
            "section": position,
            "position": position,
            "token_count": self._token_count(" ".join(units)),
        }
        return Chunk(text=" ".join(units), metadata=metadata)

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))


class FixedSizeChunkingStrategy(SemanticChunkingStrategy):
    """Chunk structured content with the same token limits and overlap rules."""


class ChunkingFactory:
    _strategies = {
        "semantic": SemanticChunkingStrategy,
        "fixed": FixedSizeChunkingStrategy,
    }

    @classmethod
    def get_strategy(
        cls, name: str = "semantic", settings: Settings | None = None
    ) -> ChunkingStrategy:
        try:
            return cls._strategies[name.lower()](settings)
        except KeyError as exc:
            raise ValueError(f"Unknown chunking strategy: {name}") from exc

    @classmethod
    def chunk(
        cls,
        document: Document,
        strategy: str = "semantic",
        settings: Settings | None = None,
    ) -> list[Chunk]:
        return cls.get_strategy(strategy, settings).chunk(document)