from fastapi.testclient import TestClient
import numpy as np

from app.main import app


client = TestClient(app)


class FakeEmbedder:
    class Provider:
        model_name = "fake-model"

    provider = Provider()

    def __init__(self, settings: object) -> None:
        del settings

    def embed_chunks(self, chunks: list[object]) -> np.ndarray:
        return np.ones((len(chunks), 3), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.ones(3, dtype=np.float32)


class FakeVectorDB:
    indexed_chunks = 0

    def __init__(self, settings: object) -> None:
        del settings

    def add_chunks(self, chunks: list[object], embeddings: np.ndarray) -> int:
        FakeVectorDB.indexed_chunks = len(chunks)
        assert embeddings.shape == (len(chunks), 3)
        return len(chunks)

    def search(self, query_vector: np.ndarray, limit: int) -> list[dict[str, object]]:
        assert query_vector.shape == (3,)
        return [
            {
                "text": "A matching chunk.",
                "metadata": {"source": "notes.txt", "position": 0},
                "distance": 0.12,
            }
        ][:limit]

    def close(self) -> None:
        pass


def test_health_check() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_endpoint_embeds_and_stores_uploaded_file(monkeypatch) -> None:
    monkeypatch.setattr("app.main.EmbeddingManager", FakeEmbedder)
    monkeypatch.setattr("app.main.VectorDBManager", FakeVectorDB)

    response = client.post(
        "/api/vector/index",
        files={"file": ("notes.txt", b"First line\nSecond line", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "indexed"
    assert body["filename"] == "notes.txt"
    assert body["chunk_count"] == body["stored_count"]
    assert body["embedding_model"] == "fake-model"
    assert FakeVectorDB.indexed_chunks == body["chunk_count"]


def test_index_endpoint_rejects_unsupported_file_type() -> None:
    response = client.post(
        "/api/vector/index",
        files={"file": ("malware.exe", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_index_endpoint_rejects_files_over_five_megabytes() -> None:
    response = client.post(
        "/api/vector/index",
        files={"file": ("large.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 413
    assert "5 MB" in response.json()["detail"]


def test_search_endpoint_embeds_query_and_returns_matches(monkeypatch) -> None:
    monkeypatch.setattr("app.main.EmbeddingManager", FakeEmbedder)
    monkeypatch.setattr("app.main.VectorDBManager", FakeVectorDB)

    response = client.post(
        "/api/search",
        json={"query": "What is in the document?", "limit": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is in the document?"
    assert body["result_count"] == 1
    assert body["results"][0]["text"] == "A matching chunk."


def test_search_endpoint_validates_query() -> None:
    response = client.post("/api/search", json={"query": "", "limit": 5})

    assert response.status_code == 422
