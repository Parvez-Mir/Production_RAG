import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.chunking import ChunkingFactory
from app.services.embeddings import EmbeddingError, EmbeddingManager
from app.services.metadata import MetadataError, MetadataManager
from app.services.parsers import (
    DocumentParseError,
    FileValidationError,
    ParserFactory,
    UnsupportedFileTypeError,
)
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


class SearchResult(BaseModel):
    text: str = Field(description="Text content of the matching document chunk.")
    metadata: dict[str, object] = Field(description="Metadata stored with the chunk.")
    distance: float | None = Field(description="Weaviate vector distance; lower is more similar.")


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResult]

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
    try:
        embedder = EmbeddingManager(settings)
        query_vector = embedder.embed_query(request.query)
        vector_db = VectorDBManager(settings)
        try:
            results = vector_db.search(query_vector, limit=request.limit)
        finally:
            vector_db.close()
        return SearchResponse(
            query=request.query,
            result_count=len(results),
            results=[SearchResult(**result) for result in results],
        )
    except (EmbeddingError, VectorDBError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
