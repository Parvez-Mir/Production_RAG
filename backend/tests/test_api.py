from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.services.vector_db import VectorDBError


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

    def get_stats(self) -> dict[str, int]:
        return {"document_count": 1, "chunk_count": 3}

    def delete_document(self, doc_id: str) -> int:
        return 1 if doc_id else 0

    def close(self) -> None:
        pass


class FakeRetriever:
    def __init__(self, vector_db: object, embeddings: object) -> None:
        del vector_db, embeddings

    def retrieve(self, query: str, top_k: int, threshold: float) -> list[object]:
        assert query == "What is in the document?"
        assert top_k == 5
        assert threshold == 0.3
        return [
            SimpleNamespace(
                text="A matching chunk.",
                metadata={"source": "notes.txt", "position": 0},
                similarity_score=0.88,
            )
        ]


class FakeMetadataManager:
    records: dict[str, dict[str, object]] = {}

    def __init__(self, settings: object) -> None:
        del settings

    def add_document(self, **document: object) -> None:
        self.records[str(document["doc_id"])] = document

    def update_document(self, doc_id: str, **changes: object) -> None:
        self.records[doc_id].update(changes)

    def get_document(self, doc_id: str) -> dict[str, object] | None:
        return self.records.get(doc_id)

    def get_stats(self) -> dict[str, int]:
        return {
            "document_count": len(self.records),
            "chunk_count": sum(int(record.get("chunk_count", 0)) for record in self.records.values()),
            "total_size_bytes": sum(int(record.get("file_size_bytes", 0)) for record in self.records.values()),
        }

    def delete_document(self, doc_id: str) -> bool:
        return self.records.pop(doc_id, None) is not None


class FailingVectorDB(FakeVectorDB):
    deleted_document_id: str | None = None

    def add_chunks(self, chunks: list[object], embeddings: np.ndarray) -> int:
        del chunks, embeddings
        raise VectorDBError("vector write failed")

    def delete_document(self, doc_id: str) -> int:
        FailingVectorDB.deleted_document_id = doc_id
        return 0


def test_health_check() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_endpoint_embeds_and_stores_uploaded_file(monkeypatch) -> None:
    monkeypatch.setattr("app.main.EmbeddingManager", FakeEmbedder)
    monkeypatch.setattr("app.main.VectorDBManager", FakeVectorDB)
    monkeypatch.setattr("app.main.MetadataManager", FakeMetadataManager)
    FakeMetadataManager.records.clear()

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
    assert FakeMetadataManager.records[body["document_id"]]["status"] == "indexed"


def test_index_endpoint_rolls_back_metadata_when_vector_storage_fails(monkeypatch) -> None:
    monkeypatch.setattr("app.main.EmbeddingManager", FakeEmbedder)
    monkeypatch.setattr("app.main.VectorDBManager", FailingVectorDB)
    monkeypatch.setattr("app.main.MetadataManager", FakeMetadataManager)
    FakeMetadataManager.records.clear()
    FailingVectorDB.deleted_document_id = None

    response = client.post(
        "/api/vector/index",
        files={"file": ("notes.txt", b"First line\nSecond line", "text/plain")},
    )

    assert response.status_code == 503
    assert FakeMetadataManager.records == {}
    assert FailingVectorDB.deleted_document_id is not None


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
    monkeypatch.setattr("app.main.RetrieverManager", FakeRetriever)

    response = client.post(
        "/api/search",
        json={"query": "What is in the document?", "limit": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is in the document?"
    assert body["result_count"] == 1
    assert body["results"][0]["text"] == "A matching chunk."
    assert body["results"][0]["similarity_score"] == 0.88
    assert body["results"][0]["distance"] == 0.12


def test_search_endpoint_validates_query() -> None:
    response = client.post("/api/search", json={"query": "", "limit": 5})

    assert response.status_code == 422


def test_stats_endpoint_returns_system_metrics(monkeypatch) -> None:
    monkeypatch.setattr("app.main.EmbeddingManager", FakeEmbedder)
    monkeypatch.setattr("app.main.VectorDBManager", FakeVectorDB)
    monkeypatch.setattr("app.main.MetadataManager", FakeMetadataManager)
    FakeMetadataManager.records.clear()
    FakeMetadataManager.records["doc-1"] = {
        "doc_id": "doc-1",
        "filename": "notes.txt",
        "file_type": "txt",
        "file_size_bytes": 120,
        "chunk_count": 2,
        "status": "indexed",
    }

    response = client.get("/api/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 1
    assert body["chunk_count"] == 3
    assert body["total_size_bytes"] == 120


def test_delete_document_endpoint_removes_document_and_chunks(monkeypatch) -> None:
    monkeypatch.setattr("app.main.EmbeddingManager", FakeEmbedder)
    monkeypatch.setattr("app.main.VectorDBManager", FakeVectorDB)
    monkeypatch.setattr("app.main.MetadataManager", FakeMetadataManager)
    FakeMetadataManager.records.clear()
    FakeMetadataManager.records["doc-1"] = {
        "doc_id": "doc-1",
        "filename": "notes.txt",
        "file_type": "txt",
        "file_size_bytes": 120,
        "chunk_count": 2,
        "status": "indexed",
    }

    response = client.delete("/api/documents/doc-1")

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert FakeMetadataManager.records == {}
