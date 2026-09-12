import logging
import uuid
from typing import Any
from urllib.parse import urlparse

import numpy as np

from app.config import Settings, get_settings
from app.services.chunking import Chunk

logger = logging.getLogger(__name__)


class VectorDBError(Exception):
    """Raised when vector database operations cannot be completed."""


class VectorDBManager:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self.client = client or self._connect()
        self.collection = self._get_or_create_collection()

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise VectorDBError(
                f"Expected one vector per chunk, got {len(chunks)} chunks and {vectors.shape}"
            )
        if not chunks:
            return 0

        inserted_ids: list[str] = []
        try:
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk_id = self._chunk_uuid(chunk)
                properties = self._properties(chunk)
                if self.collection.data.exists(uuid=chunk_id):
                    self.collection.data.replace(
                        uuid=chunk_id,
                        properties=properties,
                        vector=vector.tolist(),
                    )
                else:
                    self.collection.data.insert(
                        uuid=chunk_id,
                        properties=properties,
                        vector=vector.tolist(),
                    )
                inserted_ids.append(chunk_id)
        except Exception as exc:
            logger.exception("Failed to store chunks; removing partial writes")
            for chunk_id in inserted_ids:
                try:
                    self.collection.data.delete_by_id(chunk_id)
                except Exception:
                    logger.exception("Failed to roll back chunk %s", chunk_id)
            raise VectorDBError("Could not store document chunks") from exc
        return len(inserted_ids)

    def delete_document(self, doc_id: str) -> int:
        result = self.collection.data.delete_many(
            where=self._filter("doc_id", doc_id),
        )
        return int(getattr(result, "successful", 0))

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: Any | None = None,
    ) -> list[dict[str, Any]]:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise VectorDBError(f"Expected a one-dimensional query vector, got {vector.shape}")

        response = self.collection.query.near_vector(
            near_vector=vector.tolist(),
            limit=limit,
            filters=filters,
            return_metadata=self._metadata_query(),
        )
        return [
            {
                "text": result.properties.get("text", ""),
                "metadata": {
                    key: value
                    for key, value in result.properties.items()
                    if key != "text"
                },
                "distance": getattr(result.metadata, "distance", None),
            }
            for result in response.objects
        ]

    def get_chunks(self) -> list[dict[str, Any]]:
        chunk_count = int(self.collection.aggregate.over_all(total_count=True).total_count or 0)
        if not chunk_count:
            return []

        response = self.collection.query.fetch_objects(
            limit=chunk_count,
            return_properties=[
                "text",
                "doc_id",
                "source",
                "file_type",
                "section",
                "position",
                "chunk_id",
            ],
        )
        return [
            {
                "text": result.properties.get("text", ""),
                "metadata": {
                    key: value
                    for key, value in result.properties.items()
                    if key != "text"
                },
            }
            for result in response.objects
        ]

    def get_stats(self) -> dict[str, int]:
        aggregate = self.collection.aggregate.over_all(total_count=True)
        chunk_count = int(aggregate.total_count or 0)
        objects = self.collection.query.fetch_objects(
            limit=chunk_count,
            return_properties=["doc_id"],
        ).objects
        document_count = len({obj.properties.get("doc_id") for obj in objects})
        return {"document_count": document_count, "chunk_count": chunk_count}

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _connect(self) -> Any:
        try:
            import weaviate
        except ImportError as exc:
            raise VectorDBError("weaviate-client is required for vector storage") from exc

        parsed = urlparse(self.settings.weaviate_url)
        if parsed.hostname is None:
            raise VectorDBError(f"Invalid Weaviate URL: {self.settings.weaviate_url}")
        if self.settings.weaviate_api_key:
            return weaviate.connect_to_custom(
                http_host=parsed.hostname,
                http_port=parsed.port or 443,
                http_secure=parsed.scheme == "https",
                grpc_host=parsed.hostname,
                grpc_port=50051,
                grpc_secure=parsed.scheme == "https",
                auth_credentials=weaviate.auth.AuthApiKey(self.settings.weaviate_api_key),
            )
        return weaviate.connect_to_local(
            host=parsed.hostname,
            port=parsed.port or 8080,
            grpc_port=50051,
        )

    def _get_or_create_collection(self) -> Any:
        name = self.settings.weaviate_collection
        if not self.client.collections.exists(name):
            from weaviate.classes.config import Configure, DataType, Property

            self.client.collections.create(
                name=name,
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="file_type", data_type=DataType.TEXT),
                    Property(name="section", data_type=DataType.INT),
                    Property(name="position", data_type=DataType.INT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                ],
            )
        return self.client.collections.get(name)

    def _properties(self, chunk: Chunk) -> dict[str, Any]:
        metadata = chunk.metadata
        return {
            "text": chunk.text,
            "doc_id": str(metadata.get("doc_id", metadata.get("source", "document"))),
            "source": str(metadata.get("source", "")),
            "file_type": str(metadata.get("file_type", "")),
            "section": int(metadata.get("section", 0)),
            "position": int(metadata.get("position", 0)),
            "chunk_id": str(metadata.get("chunk_id", "")),
        }

    def _chunk_uuid(self, chunk: Chunk) -> str:
        chunk_id = str(chunk.metadata.get("chunk_id", ""))
        if not chunk_id:
            raise VectorDBError("Every chunk must have a chunk_id")
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _filter(self, property_name: str, value: str) -> Any:
        from weaviate.classes.query import Filter

        return Filter.by_property(property_name).equal(value)

    def _metadata_query(self) -> Any:
        from weaviate.classes.query import MetadataQuery

        return MetadataQuery(distance=True)