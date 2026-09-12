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

Index a document, up to 5 MB:

```bash
curl -X POST http://localhost:8000/api/vector/index \
	-F "file=@path/to/document.txt"
```

The endpoint parses the file, creates chunks and embeddings, then stores them in
Weaviate. Supported extensions are `.pdf`, `.txt`, `.md`, `.csv`, and `.json`.

Search the indexed chunks:

```bash
curl -X POST http://localhost:8000/api/search \
	-H "Content-Type: application/json" \
	-d '{"query":"What is this document about?","limit":5}'
```

The query is embedded with the same model used for document chunks. Weaviate
returns the closest matching chunks through the LangChain Weaviate retriever.
Results include metadata, normalized similarity scores, and the legacy vector
distance field. Matches below the default similarity threshold of `0.3` are
filtered out.

View the current system stats:

```bash
curl http://localhost:8000/api/stats
```

Delete a document and its stored vectors:

```bash
curl -X DELETE http://localhost:8000/api/documents/<doc_id>
```

Document metadata is persisted at the configured `METADATA_FILE_PATH` (default:
`app/data/metadata.json`). Each record includes the document ID, filename, file
type, upload time, file size, chunk count, and indexing status. Metadata writes
are atomic, and failed indexing removes both metadata and any stored vectors.

Interactive API docs: `http://localhost:8000/docs`

## LLM Generation

Task 7 provides Gemini generation with an optional Ollama fallback through the
same client interface. Configure the provider in `.env` with `LLM_PROVIDER=auto`
to use Gemini when `GEMINI_API_KEY` is present and fall back to Ollama when
Gemini fails. Set `LLM_PROVIDER=gemini`, `LLM_PROVIDER=claude`, or
`LLM_PROVIDER=ollama` to force one provider. For local fallback, install Ollama,
start it, and pull the configured model (default: `llama3.1:8b`).

The default Gemini model is `gemini-3.6-flash`; generation uses 1024 max tokens,
temperature `0.7`, a 30-second timeout, and two retries. Provider calls return
generated text, token usage when available, latency, model, and provider metadata.
Keep `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` only in the local `.env`; never
commit them.

## Tests

```bash
pytest
```

## Document Parsing

Supported formats are PDF, TXT, Markdown, CSV, and JSON. Files are validated against
the configured extension allowlist and maximum size before indexing.

```python
from app.services.parsers import ParserFactory

document = ParserFactory.parse("path/to/document.pdf")
print(document.content)
print(document.metadata)
```

The parser returns a common `Document` object with `content` and `metadata`, ready
for chunking and embedding.

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
