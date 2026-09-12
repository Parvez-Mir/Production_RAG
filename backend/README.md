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

Interactive API docs: `http://localhost:8000/docs`

## Tests

```bash
pytest
```
