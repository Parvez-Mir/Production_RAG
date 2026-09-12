# RAG Backend

FastAPI backend for the RAG document ingestion project.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Start the local Weaviate vector database from the project root:

```bash
docker compose up -d
```

Weaviate is available at `http://localhost:8080` with persistent data stored in
the `weaviate_data` Docker volume. The vector database uses self-provided vectors
from the configured embedding provider; it does not download or generate vectors
itself.

Stop it with:

```bash
docker compose down
```

Health check: `GET http://localhost:8000/api/health`

Upload and parse a document, up to 5 MB:

```bash
curl -X POST http://localhost:8000/api/ingest \
	-F "file=@path/to/document.txt"
```

The response includes the parsed content and metadata. Supported extensions are
`.pdf`, `.txt`, `.md`, `.csv`, and `.json`.

Interactive API docs: `http://localhost:8000/docs`

## Tests

```bash
pytest
```

## Document Parsing

Supported formats are PDF, TXT, Markdown, CSV, and JSON. Files are validated against
the configured extension allowlist and maximum size before parsing.

```python
from app.services.parsers import ParserFactory

document = ParserFactory.parse("path/to/document.pdf")
print(document.content)
print(document.metadata)
```

The parser returns a common `Document` object with `content` and `metadata`, ready
for the chunking stage.

To inspect chunking through the API while developing, start the server and upload
a document to the temporary debug endpoint:

```bash
curl -X POST http://localhost:8000/api/debug/chunk \
	-F "file=@path/to/document.txt"
```

The response includes `chunk_count` and each chunk's text and metadata. This route
is intended for local development and inspection, not production use.

## Chunking

Parsed documents can be split into semantic, token-aware chunks using the configured
chunk size and overlap:

```python
from app.services.chunking import ChunkingFactory

chunks = ChunkingFactory.chunk(document)
```

The default strategy preserves sentence boundaries where possible, carries safe
sentence overlap between chunks, and adds chunk identifiers, positions, and token
counts to the document metadata.
