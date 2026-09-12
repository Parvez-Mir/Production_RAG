import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings


class MetadataError(Exception):
    """Raised when document metadata cannot be read or persisted."""


@dataclass
class DocumentMetadata:
    doc_id: str
    filename: str
    file_type: str
    upload_time: str
    chunk_count: int
    file_size_bytes: int
    status: str


class MetadataManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(self.settings.metadata_file_path)

    def add_document(
        self,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        chunk_count: int = 0,
        status: str = "indexing",
        doc_id: str | None = None,
    ) -> DocumentMetadata:
        document = DocumentMetadata(
            doc_id=doc_id or str(uuid.uuid4()),
            filename=filename,
            file_type=file_type,
            upload_time=datetime.now(timezone.utc).isoformat(),
            chunk_count=chunk_count,
            file_size_bytes=file_size_bytes,
            status=status,
        )
        documents = self.list_documents()
        documents.append(document)
        self._save(documents)
        return document

    def get_document(self, doc_id: str) -> DocumentMetadata | None:
        return next((document for document in self.list_documents() if document.doc_id == doc_id), None)

    def get_by_filename(self, filename: str) -> list[DocumentMetadata]:
        return [document for document in self.list_documents() if document.filename == filename]

    def list_documents(self) -> list[DocumentMetadata]:
        payload = self._load()
        documents = payload.get("documents", [])
        if not isinstance(documents, list):
            raise MetadataError("Metadata file has an invalid documents value")
        try:
            return [DocumentMetadata(**document) for document in documents]
        except (TypeError, ValueError) as exc:
            raise MetadataError("Metadata file contains an invalid document record") from exc

    def update_document(self, doc_id: str, **changes: Any) -> DocumentMetadata:
        documents = self.list_documents()
        for index, document in enumerate(documents):
            if document.doc_id == doc_id:
                allowed_changes = {
                    key: value for key, value in changes.items() if key in DocumentMetadata.__annotations__
                }
                updated = DocumentMetadata(**{**asdict(document), **allowed_changes})
                documents[index] = updated
                self._save(documents)
                return updated
        raise MetadataError(f"Document not found: {doc_id}")

    def delete_document(self, doc_id: str) -> bool:
        documents = self.list_documents()
        remaining = [document for document in documents if document.doc_id != doc_id]
        if len(remaining) == len(documents):
            return False
        self._save(remaining)
        return True

    def get_stats(self) -> dict[str, int]:
        documents = self.list_documents()
        return {
            "document_count": len(documents),
            "chunk_count": sum(document.chunk_count for document in documents),
            "total_size_bytes": sum(document.file_size_bytes for document in documents),
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"documents": []}
        try:
            with self.path.open(encoding="utf-8") as metadata_file:
                payload = json.load(metadata_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetadataError(f"Could not read metadata file: {self.path}") from exc
        if not isinstance(payload, dict):
            raise MetadataError("Metadata file must contain a JSON object")
        return payload

    def _save(self, documents: list[DocumentMetadata]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": [asdict(document) for document in documents]}
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                json.dump(payload, temporary_file, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            raise MetadataError(f"Could not write metadata file: {self.path}") from exc
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)