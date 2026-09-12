# RAG System Progress Tracker

Last updated: 2026-09-12
Current phase: Phase 1 - Foundation & Document Ingestion
Overall status: In progress

## Current Position

- Planning documents are aligned on Weaviate instead of ChromaDB.
- Minimal boilerplate backend structure has been created; feature-specific folders will be added when needed.
- Task 1 is in progress; the next step is validating the application startup path.
- No document parsing, embedding, or vector database behavior has been implemented yet.

## Phase 1 Tasks

| Task | Status | Notes |
| --- | --- | --- |
| 1. Project setup and configuration | In progress | Minimal runtime structure, configuration, logging, health route, and test created. |
| 2. Multi-format document parsing | Complete | Parser factory supports PDF, TXT, MD, CSV, and JSON with validation and tests. |
| 3. Intelligent chunking | Complete | Semantic token-aware chunks with sentence-safe overlap and metadata. |
| 4. Embedding generation | Complete | Provider-based embedding service with configurable model, batching, device, normalization, and injected test providers. |
| 5. Weaviate vector database | Complete | Local Docker Weaviate manager with self-provided vectors, upsert, search, deletion, stats, and mocked tests. |
| 6. Metadata tracking | Complete | Atomic JSON metadata store with CRUD, stats, persistence, and ingestion rollback. |
| 7. Ingestion orchestration pipeline | Not started | Coordinate parse, chunk, embed, store, metadata, and rollback. |
| 8. FastAPI endpoints | Not started | Add ingest, stats, health, and document deletion routes. |
| 9. Testing strategy | Not started | Add unit and integration coverage for Phase 1. |
| 10. Documentation | In progress | Initial backend setup documentation created. |

## Acceptance Criteria Progress

- [x] Upload and parse PDF, TXT, MD, CSV, and JSON files (5 MB limit).
- [x] Parse PDF, TXT, MD, CSV, and JSON files through the parser service.
- [x] Produce semantic chunks with configured token limits and overlap.
- [x] Generate model-dimension embeddings with configurable `all-MiniLM-L6-v2` provider.
- [x] Store vectors and metadata in local Weaviate.
- [x] Track documents and statistics in `metadata.json`.
- [ ] Expose `/api/stats` and complete the remaining API endpoints.
- [ ] Support document deletion and rollback after failed ingestion.
- [ ] Pass at least 5 unit tests plus integration coverage.
- [ ] Provide setup and API documentation.

## Known Issues / Follow-up

- [ ] **Endpoint latency:** `/api/vector/index` and `/api/search` each take roughly 6 seconds for a two-page PDF or query because the embedding model and Weaviate connection are initialized per request. Reuse long-lived service instances after the current API behavior is stable.
- [ ] **Search relevance:** Queries do not consistently return the most relevant chunks. Investigate chunk size and overlap, document-aware chunk IDs, query/document embedding consistency, and keyword or reranking support.
- [ ] **Performance measurement:** Add timing for parsing, chunking, embedding, and Weaviate operations so latency improvements can be measured separately.

## Update Log

| Date | Update |
| --- | --- |
| 2026-09-12 | Created the Phase 1 tracker and backend boilerplate. |
| 2026-09-12 | Simplified the boilerplate by removing unused package folders, `.gitkeep`, and generated `__pycache__` directories. |
| 2026-09-12 | Implemented Task 2 parsers, validation errors, dependencies, and 13 parser tests on `feat/document_parser`. |
| 2026-09-12 | Added `/api/ingest` with 5 MB streaming upload validation and API tests. |
| 2026-09-12 | Implemented Task 3 token-aware semantic chunking with overlap, metadata, strategy factory, and tests. |
| 2026-09-12 | Implemented Task 4 modular embedding providers, sentence-transformers integration, configuration, and tests. |
| 2026-09-12 | Implemented Task 5 Weaviate vector storage manager, local Docker configuration, and mocked tests. |
| 2026-09-12 | Recorded endpoint latency and search relevance issues for later investigation. |
| 2026-09-12 | Implemented Task 6 JSON metadata tracking, atomic persistence, document stats, and indexing rollback tests. |
