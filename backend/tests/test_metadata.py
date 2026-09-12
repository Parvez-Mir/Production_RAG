import json

import pytest

from app.config import Settings
from app.services.metadata import MetadataError, MetadataManager


def make_manager(tmp_path) -> MetadataManager:
    return MetadataManager(Settings(metadata_file_path=str(tmp_path / "metadata.json")))


def test_add_get_and_persist_document(tmp_path) -> None:
    manager = make_manager(tmp_path)

    created = manager.add_document("report.pdf", "pdf", 2400, chunk_count=4, status="indexed")
    reloaded = make_manager(tmp_path).get_document(created.doc_id)

    assert reloaded == created
    assert created.upload_time.endswith("+00:00")


def test_update_delete_and_lookup_by_filename(tmp_path) -> None:
    manager = make_manager(tmp_path)
    created = manager.add_document("notes.txt", "txt", 120)

    updated = manager.update_document(created.doc_id, chunk_count=3, status="indexed")

    assert updated.chunk_count == 3
    assert manager.get_by_filename("notes.txt") == [updated]
    assert manager.delete_document(created.doc_id) is True
    assert manager.get_document(created.doc_id) is None
    assert manager.delete_document(created.doc_id) is False


def test_stats_include_document_chunks_and_size(tmp_path) -> None:
    manager = make_manager(tmp_path)
    manager.add_document("one.txt", "txt", 100, chunk_count=2)
    manager.add_document("two.json", "json", 300, chunk_count=5)

    assert manager.get_stats() == {
        "document_count": 2,
        "chunk_count": 7,
        "total_size_bytes": 400,
    }


def test_missing_file_starts_empty_and_invalid_file_raises(tmp_path) -> None:
    manager = make_manager(tmp_path)
    assert manager.list_documents() == []

    manager.path.write_text(json.dumps({"documents": "invalid"}), encoding="utf-8")
    with pytest.raises(MetadataError, match="invalid documents"):
        manager.list_documents()