import numpy as np
import pytest

from app.config import Settings
from app.services.chunking import Chunk
from app.services.embeddings import EmbeddingError, EmbeddingManager
from app.services.embeddings.sentence_transformer import SentenceTransformerProvider


class FakeProvider:
    dimension = 3
    model_name = "fake-model"

    def __init__(self) -> None:
        self.text_batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.text_batches.append(texts)
        return np.array([[len(text), 1, 0] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([len(text), 1, 0], dtype=np.float32)


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((texts, int(kwargs["batch_size"])))
        return np.ones((len(texts), 2), dtype=np.float32)


def test_embedding_manager_embeds_chunks_with_injected_provider() -> None:
    provider = FakeProvider()
    manager = EmbeddingManager(Settings(), provider=provider)
    chunks = [Chunk(text="one", metadata={}), Chunk(text="two words", metadata={})]

    vectors = manager.embed_chunks(chunks)

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert provider.text_batches == [["one", "two words"]]


def test_embedding_manager_supports_query_embeddings_and_model_info() -> None:
    manager = EmbeddingManager(Settings(), provider=FakeProvider())

    assert manager.embed_query("hello").shape == (3,)
    assert manager.model_info() == {
        "provider": "sentence-transformers",
        "model": "fake-model",
        "dimension": 3,
    }


def test_embedding_manager_returns_empty_matrix_for_empty_chunks() -> None:
    manager = EmbeddingManager(Settings(), provider=FakeProvider())

    vectors = manager.embed_chunks([])

    assert vectors.shape == (0, 3)


def test_embedding_manager_rejects_empty_queries() -> None:
    manager = EmbeddingManager(Settings(), provider=FakeProvider())

    with pytest.raises(EmbeddingError, match="empty query"):
        manager.embed_query(" ")


def test_sentence_transformer_provider_batches_using_configured_size() -> None:
    model = FakeSentenceTransformer()
    settings = Settings(embedding_batch_size=2, embedding_model_name="custom-model")
    provider = SentenceTransformerProvider(settings, model=model)

    vectors = provider.embed_texts(["one", "two", "three", "four", "five"])

    assert provider.dimension == 2
    assert provider.model_name == "custom-model"
    assert vectors.shape == (5, 2)
    assert model.calls == [(["one", "two"], 2), (["three", "four"], 2), (["five"], 2)]