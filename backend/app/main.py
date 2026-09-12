import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.services.chunking import ChunkingFactory
from app.services.parsers import (
    DocumentParseError,
    FileValidationError,
    ParserFactory,
    UnsupportedFileTypeError,
)
from app.utils.logger import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
MAX_UPLOAD_SIZE_MB = 5

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


@app.post("/api/ingest", tags=["documents"])
async def ingest_file(file: UploadFile = File(...)) -> dict[str, object]:
    """Upload and parse one document without storing it yet."""
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
    total_bytes = 0
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
        return {
            "status": "parsed",
            "filename": file.filename,
            "size_bytes": total_bytes,
            "content": document.content,
            "metadata": {**document.metadata, "source": file.filename},
        }
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except FileValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()
        if temporary_path:
            os.unlink(temporary_path)


@app.post("/api/debug/chunk", tags=["debug"])
async def debug_chunk_file(file: UploadFile = File(...)) -> dict[str, object]:
    """Parse an uploaded document and return its chunks for inspection."""
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
    total_bytes = 0
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
        document.metadata["source"] = file.filename
        chunks = ChunkingFactory.chunk(document, settings=settings)

        return {
            "status": "chunked",
            "filename": file.filename,
            "size_bytes": total_bytes,
            "chunk_count": len(chunks),
            "chunks": [
                {"text": chunk.text, "metadata": chunk.metadata}
                for chunk in chunks
            ],
        }
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except FileValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()
        if temporary_path:
            os.unlink(temporary_path)
