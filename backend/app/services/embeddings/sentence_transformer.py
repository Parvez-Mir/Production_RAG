import logging
from typing import Any

import numpy as np

from app.config import Settings
from app.services.embeddings.base import EmbeddingError

logger = logging.getLogger(__name__)


class SentenceTransformerProvider:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is required for the configured embedding provider"
                ) from exc

            model = SentenceTransformer(
                settings.embedding_model_name,
                device=settings.embedding_device,
            )

        self._model = model
        self._model_name = settings.embedding_model_name
        self._batch_size = settings.embedding_batch_size
        self._normalize = settings.embedding_normalize
        self._dimension = model.get_sentence_embedding_dimension()
        if not self._dimension:
            raise EmbeddingError("The embedding model did not report a vector dimension")

    @property
    def dimension(self) -> int:
        return int(self._dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors: list[np.ndarray] = []
        try:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                logger.debug("Embedding batch %d-%d", start + 1, start + len(batch))
                encoded = self._model.encode(
                    batch,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                vectors.append(self._validate_vectors(np.asarray(encoded), len(batch)))
        except (MemoryError, RuntimeError) as exc:
            raise EmbeddingError("The embedding model could not process the requested batch") from exc

        return np.vstack(vectors).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_texts([text])
        return vectors[0]

    def _validate_vectors(self, vectors: np.ndarray, expected_count: int) -> np.ndarray:
        if vectors.ndim != 2 or vectors.shape != (expected_count, self.dimension):
            raise EmbeddingError(
                f"Expected ({expected_count}, {self.dimension}) embeddings, got {vectors.shape}"
            )
        return vectors