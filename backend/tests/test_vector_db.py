from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.services.chunking import Chunk
from app.services.vector_db import VectorDBError, VectorDBManager


def make_chunk(chunk_id: str, doc_id: str = "document-1") -> Chunk:
    return Chunk(
        text=f"content for {chunk_id}",
        metadata={
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source": "notes.txt",
            "file_type": "txt",
            "section": 0,
            "position": 0,
        },
    )


class FakeData:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.deleted: list[str] = []

    def exists(self, uuid: str) -> bool:
        return uuid in self.objects

    def insert(self, uuid: str, properties: dict[str, object], vector: list[float]) -> None:
        self.objects[uuid] = {"properties": properties, "vector": vector}

    def replace(self, uuid: str, properties: dict[str, object], vector: list[float]) -> None:
        self.objects[uuid] = {"properties": properties, "vector": vector}

    def delete_by_id(self, uuid: str) -> None:
        self.deleted.append(uuid)
        self.objects.pop(uuid, None)

    def delete_many(self, where: object) -> SimpleNamespace:
        del where
        count = len(self.objects)
        self.objects.clear()
        return SimpleNamespace(successful=count)


class FakeCollection:
    def __init__(self) -> None:
        self.data = FakeData()
        self.aggregate = SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=len(self.data.objects))
        )
        self.query = SimpleNamespace(
            near_vector=lambda **kwargs: SimpleNamespace(
                objects=[
                    SimpleNamespace(
                        properties={"text": "matched", "doc_id": "document-1"},
                        metadata=SimpleNamespace(distance=0.12),
                    )
                ]
            ),
            fetch_objects=lambda **kwargs: SimpleNamespace(
                objects=[
                    SimpleNamespace(properties=record["properties"])
                    for record in self.data.objects.values()
                ]
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        collection = FakeCollection()
        self.collections = SimpleNamespace(
            exists=lambda name: True,
            get=lambda name: collection,
        )

    def close(self) -> None:
        pass


def test_add_chunks_upserts_vectors_and_stats() -> None:
    manager = VectorDBManager(Settings(), client=FakeClient())
    chunk = make_chunk("chunk-1")

    assert manager.add_chunks([chunk], np.array([[1.0, 2.0]], dtype=np.float32)) == 1
    assert manager.add_chunks([chunk], np.array([[3.0, 4.0]], dtype=np.float32)) == 1
    assert manager.get_stats() == {"document_count": 1, "chunk_count": 1}


def test_add_chunks_requires_one_vector_per_chunk() -> None:
    manager = VectorDBManager(Settings(), client=FakeClient())

    with pytest.raises(VectorDBError, match="one vector per chunk"):
        manager.add_chunks([make_chunk("chunk-1")], np.empty((0, 2)))


def test_search_returns_text_metadata_and_distance() -> None:
    manager = VectorDBManager(Settings(), client=FakeClient())

    results = manager.search(np.array([1.0, 2.0]), limit=3)

    assert results == [
        {
            "text": "matched",
            "metadata": {"doc_id": "document-1"},
            "distance": 0.12,
        }
    ]


def test_delete_document_reports_deleted_chunks() -> None:
    manager = VectorDBManager(Settings(), client=FakeClient())

    assert manager.delete_document("document-1") == 0