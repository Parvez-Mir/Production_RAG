# RAG System Progress Tracker

Last updated: 2026-09-12
Current phase: Phase 2 - Retrieval Pipeline & LLM Integration
Overall status: In progress

## Current Position

- The Phase 1 backend foundation is complete and validated.
- Parsing, chunking, embedding, metadata persistence, vector storage, and the remaining API routes are implemented.
- Phase 2 planning is complete; implementation will cover retrieval, generation, chat orchestration, and resilience.
- Full integration testing is out of scope for this phase; component-level checks remain part of implementation work.

## Phase 1 Tasks

| Task | Status | Notes |
| --- | --- | --- |
| 1. Project setup and configuration | Complete | Runtime structure, settings, logging, health route, and backend configuration are in place. |
| 2. Multi-format document parsing | Complete | Parser factory supports PDF, TXT, MD, CSV, and JSON with validation and tests. |
| 3. Intelligent chunking | Complete | Semantic token-aware chunks with sentence-safe overlap and metadata. |
| 4. Embedding generation | Complete | Provider-based embedding service with configurable model, batching, device, normalization, and injected test providers. |
| 5. Weaviate vector database | Complete | Local Docker Weaviate manager with self-provided vectors, upsert, search, deletion, stats, and mocked tests. |
| 6. Metadata tracking | Complete | Atomic JSON metadata store with CRUD, stats, persistence, and ingestion rollback. |
| 7. Ingestion orchestration pipeline | Complete | Upload pipeline coordinates parse, chunk, embed, store, metadata updates, and rollback on failure. |
| 8. FastAPI endpoints | Complete | `/api/vector/index`, `/api/search`, `/api/stats`, and document deletion route are implemented. |
| 9. Testing strategy | Complete | Unit and API tests cover chunking, parsing, embeddings, vector storage, metadata, and endpoints. |
| 10. Documentation | Complete | Backend setup and API usage are documented in the README and tracker. |

## Acceptance Criteria Progress

- [x] Upload and parse PDF, TXT, MD, CSV, and JSON files (5 MB limit).
- [x] Parse PDF, TXT, MD, CSV, and JSON files through the parser service.
- [x] Produce semantic chunks with configured token limits and overlap.
- [x] Generate model-dimension embeddings with configurable `all-MiniLM-L6-v2` provider.
- [x] Store vectors and metadata in local Weaviate.
- [x] Track documents and statistics in `metadata.json`.
- [x] Expose `/api/stats` and complete the remaining API endpoints.
- [x] Support document deletion and rollback after failed ingestion.
- [x] Pass at least 5 unit tests plus integration coverage.
- [x] Provide setup and API documentation.

## Known Issues / Follow-up

- [ ] **Endpoint latency:** `/api/vector/index` and `/api/search` each take roughly 6 seconds for a two-page PDF or query because the embedding model and Weaviate connection are initialized per request. Reuse long-lived service instances after the current API behavior is stable.
- [ ] **Search relevance:** Queries do not consistently return the most relevant chunks. Investigate chunk size and overlap, document-aware chunk IDs, query/document embedding consistency, and keyword or reranking support.
- [ ] **Performance measurement:** Add timing for parsing, chunking, embedding, and Weaviate operations so latency improvements can be measured separately.

## Phase 2 Scope

| Area | Status | Notes |
| --- | --- | --- |
| Retrieval pipeline | Task 1 complete | LangChain vector retrieval over Weaviate is implemented; BM25, hybrid retrieval, and optional re-ranking remain. |
| Context and prompts | Planned | Context assembly and versioned prompt templates. |
| LLM integration | Planned | Claude client with Ollama fallback and resilience handling. |
| Chat endpoint | Planned | `/api/chat` response with sources and timing metadata. |
| Full integration testing | Out of scope | No dedicated end-to-end integration test suite in Phase 2. |

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
| 2026-09-12 | Added `/api/stats` and document deletion endpoints, plus regression coverage for the API contract and docs refresh. |
| 2026-09-12 | Defined Phase 2 scope and removed dedicated integration testing from the Phase 2 plan. |
| 2026-09-12 | Completed Phase 2 Task 1: LangChain vector retrieval over Weaviate with thresholds, metadata filters, normalized scores, and focused tests. |
