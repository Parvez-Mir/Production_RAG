import os
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.chunking import ChunkingFactory
from app.services.embeddings import EmbeddingError, EmbeddingManager
from app.services.bm25 import BM25SearchManager
from app.services.hybrid_retrieval import HybridRetrievalError, HybridRetriever
from app.services.metadata import MetadataError, MetadataManager
from app.services.parsers import (
    DocumentParseError,
    FileValidationError,
    ParserFactory,
    UnsupportedFileTypeError,
)
from app.services.retrieval import RetrievedChunk, RetrievalError, RetrieverManager
from app.services.vector_db import VectorDBError, VectorDBManager
from app.utils.logger import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
MAX_UPLOAD_SIZE_MB = settings.max_file_size_mb


class IndexResponse(BaseModel):
    status: str = Field(description="Indexing result status.")
    document_id: str = Field(description="Identifier assigned to the uploaded document.")
    filename: str = Field(description="Original uploaded filename.")
    chunk_count: int = Field(description="Number of chunks created from the document.")
    stored_count: int = Field(description="Number of chunk vectors stored in Weaviate.")
    embedding_model: str = Field(description="Embedding model used for the document.")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language question or search phrase.")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of matches to return.")
    retrieval_mode: Literal["hybrid", "vector_only"] = Field(
        default="hybrid",
        description="Retrieval strategy used for this search.",
    )


class SearchResult(BaseModel):
    text: str = Field(description="Text content of the matching document chunk.")
    metadata: dict[str, object] = Field(description="Metadata stored with the chunk.")
    distance: float | None = Field(description="Weaviate vector distance; lower is more similar.")
    similarity_score: float = Field(description="Normalized relevance score from 0 to 1.")


class SearchResponse(BaseModel):
    query: str
    retrieval_mode: str
    result_count: int
    results: list[SearchResult]


class StatsResponse(BaseModel):
    document_count: int = Field(description="Number of documents currently tracked.")
    chunk_count: int = Field(description="Number of indexed chunks in the vector database.")
    total_size_bytes: int = Field(description="Combined size of all tracked documents in bytes.")


class DeleteDocumentResponse(BaseModel):
    status: str = Field(description="Deletion status for the document.")
    document_id: str = Field(description="Identifier of the deleted document.")


app = FastAPI(
    title="RAG Backend",
    description="Document ingestion and retrieval backend.",
    version="0.1.0",
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/vector/index",
    response_model=IndexResponse,
    tags=["vector search"],
    summary="Index an uploaded document",
    description=(
        "Upload a supported document, split it into chunks, generate embeddings, "
        "and store the chunks and vectors in local Weaviate."
    ),
)
async def index_file(file: UploadFile = File(...)) -> IndexResponse:
    """Index an uploaded document for later semantic search."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    settings = get_settings()
    suffix = Path(file.filename).suffix.lower()
    if not suffix:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file must have a supported extension.",
        )

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    temporary_path: str | None = None
    vector_db: VectorDBManager | None = None
    metadata_manager = MetadataManager(settings)
    metadata_created = False
    total_bytes = 0
    document_id = str(uuid.uuid4())

    def rollback() -> None:
        if vector_db is not None:
            try:
                vector_db.delete_document(document_id)
            except VectorDBError:
                pass
        if metadata_created:
            try:
                metadata_manager.delete_document(document_id)
            except MetadataError:
                pass

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.",
                    )
                temporary_file.write(chunk)

        document = ParserFactory.parse(temporary_path, settings=settings)
        document.metadata.update({"source": file.filename, "doc_id": document_id})
        metadata_manager.add_document(
            filename=file.filename,
            file_type=suffix.lstrip("."),
            file_size_bytes=total_bytes,
            doc_id=document_id,
        )
        metadata_created = True
        chunks = ChunkingFactory.chunk(document, settings=settings)
        embedder = EmbeddingManager(settings)
        embeddings = embedder.embed_chunks(chunks)
        vector_db = VectorDBManager(settings)
        stored_count = vector_db.add_chunks(chunks, embeddings)
        metadata_manager.update_document(
            document_id,
            chunk_count=len(chunks),
            status="indexed",
        )
        return IndexResponse(
            status="indexed",
            document_id=document_id,
            filename=file.filename,
            chunk_count=len(chunks),
            stored_count=stored_count,
            embedding_model=embedder.provider.model_name,
        )
    except HTTPException:
        rollback()
        raise
    except UnsupportedFileTypeError as exc:
        rollback()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except FileValidationError as exc:
        rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentParseError as exc:
        rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (EmbeddingError, MetadataError, VectorDBError) as exc:
        rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    finally:
        if vector_db is not None:
            vector_db.close()
        await file.close()
        if temporary_path:
            os.unlink(temporary_path)

@app.post(
    "/api/search",
    response_model=SearchResponse,
    tags=["vector search"],
    summary="Search indexed document chunks",
    description=(
        "Embed a natural-language query and find the most similar indexed chunks "
        "in local Weaviate."
    ),
)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Return document chunks nearest to the embedded query."""
    settings = get_settings()
    vector_db: VectorDBManager | None = None
    try:
        embedder = EmbeddingManager(settings)
        vector_db = VectorDBManager(settings)
        retriever = RetrieverManager(vector_db=vector_db, embeddings=embedder)
        if request.retrieval_mode == "vector_only":
            retrieved_chunks = retriever.retrieve(
                request.query,
                top_k=request.limit,
                threshold=0.3,
            )
        else:
            keyword_chunks = [
                RetrievedChunk(
                    chunk_id=str(chunk["metadata"].get("chunk_id", "")),
                    text=str(chunk["text"]),
                    similarity_score=0.0,
                    metadata=dict(chunk["metadata"]),
                    source_document=str(
                        chunk["metadata"].get(
                            "source", chunk["metadata"].get("doc_id", "unknown")
                        )
                    ),
                )
                for chunk in vector_db.get_chunks()
                if chunk["metadata"].get("chunk_id")
            ]
            hybrid_retriever = HybridRetriever(retriever, BM25SearchManager(keyword_chunks))
            retrieved_chunks = hybrid_retriever.retrieve(request.query, top_k=request.limit)
        return SearchResponse(
            query=request.query,
            retrieval_mode=request.retrieval_mode,
            result_count=len(retrieved_chunks),
            results=[
                SearchResult(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    distance=1.0 - chunk.similarity_score,
                    similarity_score=chunk.similarity_score,
                )
                for chunk in retrieved_chunks
            ],
        )
    except (EmbeddingError, RetrievalError, HybridRetrievalError, VectorDBError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    finally:
        if vector_db is not None:
            vector_db.close()


@app.get("/api/stats", response_model=StatsResponse, tags=["system"], summary="Get document and chunk statistics")
def get_stats() -> StatsResponse:
    """Return metadata and vector store counts for tracked documents."""
    settings = get_settings()
    metadata_manager = MetadataManager(settings)
    try:
        documented_stats = metadata_manager.get_stats()
        vector_db = VectorDBManager(settings)
        try:
            vector_stats = vector_db.get_stats()
        finally:
            vector_db.close()
    except (MetadataError, VectorDBError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    merged = dict(documented_stats)
    for key in ("document_count", "chunk_count"):
        if key in vector_stats:
            merged[key] = vector_stats[key]
    return StatsResponse(**merged)


@app.delete(
    "/api/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    tags=["system"],
    summary="Delete a document and its indexed chunks",
)
def delete_document(document_id: str) -> DeleteDocumentResponse:
    """Remove a document from both the metadata store and the vector database."""
    settings = get_settings()
    metadata_manager = MetadataManager(settings)
    try:
        if not metadata_manager.get_document(document_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        vector_db = VectorDBManager(settings)
        try:
            vector_db.delete_document(document_id)
        finally:
            vector_db.close()

        deleted = metadata_manager.delete_document(document_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )
    except HTTPException:
        raise
    except (MetadataError, VectorDBError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return DeleteDocumentResponse(status="deleted", document_id=document_id)
