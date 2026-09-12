from typing import Protocol

import numpy as np


class EmbeddingError(Exception):
    """Raised when embedding generation cannot be completed."""


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...