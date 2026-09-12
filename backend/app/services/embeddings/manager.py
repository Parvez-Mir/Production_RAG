import logging

import numpy as np

from app.config import Settings, get_settings
from app.services.chunking import Chunk
from app.services.embeddings.base import EmbeddingError, EmbeddingProvider
from app.services.embeddings.factory import EmbeddingFactory

logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = EmbeddingFactory.create(self.settings, provider=provider)

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise EmbeddingError("Cannot embed an empty query")
        vector = np.asarray(self.provider.embed_query(text), dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise EmbeddingError(
                f"Expected query embedding shape ({self.dimension},), got {vector.shape}"
            )
        return vector

    def model_info(self) -> dict[str, str | int]:
        return {
            "provider": self.settings.embedding_provider,
            "model": self.provider.model_name,
            "dimension": self.dimension,
        }

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = np.asarray(self.provider.embed_texts(texts), dtype=np.float32)
        expected_shape = (len(texts), self.dimension)
        if vectors.shape != expected_shape:
            raise EmbeddingError(
                f"Expected embedding shape {expected_shape}, got {vectors.shape}"
            )
        logger.info("Generated %d embeddings with %s", len(texts), self.provider.model_name)
        return vectors