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
