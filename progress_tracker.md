# RAG System Progress Tracker

Last updated: 2026-09-12
Current phase: Phase 2 - Retrieval Pipeline & LLM Integration
Overall status: In progress

## Current Position

- **Phase 1 - Foundation and ingestion:** complete and validated. This phase takes documents in, parses and chunks them, creates embeddings, stores vectors, and tracks document metadata.
- **Phase 2 - Retrieval and answer generation:** in progress. This phase takes a user query, retrieves and ranks relevant chunks, builds prompts, calls an LLM, and returns a cited answer.
- Full integration testing is out of scope for this phase; component-level checks remain part of implementation work.

## Phase Boundaries

| Phase | Human-readable responsibility | Main output |
| --- | --- | --- |
| Phase 1 | Prepare the knowledge base by ingesting and indexing source documents. | Searchable document chunks with embeddings and metadata. |
| Phase 2 | Use the knowledge base to answer a user's question accurately and with citations. | Ranked context, generated answer, sources, and response metrics. |

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
- [ ] **LLM streaming:** Add optional streaming responses for interactive clients.
- [ ] **LLM cost tracking:** Persist token usage and provider pricing estimates for monitoring.
- [ ] **LLM health checks:** Check provider availability and configured model support before generation.
- [ ] **Selective LLM retries:** Retry only transient provider failures instead of all unexpected errors.

## Phase 2 Tasks

| Task | Status | Human-readable scope | Notes |
| --- | --- | --- | --- |
| 1. Vector similarity search | Complete | Find semantically similar chunks in Weaviate for a user query. | LangChain retrieval, thresholds, metadata filters, normalized scores, and tests are implemented. |
| 2. BM25 keyword search | Complete | Find chunks that contain important exact keywords or phrases. | BM25 indexing, scoring, normalization, dynamic updates, and tests are implemented. |
| 3. Hybrid retrieval | Complete | Combine semantic and keyword results into one deduplicated ranking. | Weighted vector/BM25 merging and tests are implemented. |
| 4. Re-ranking pipeline | Complete | Improve the initial ranking using query-to-chunk relevance scoring. | Cross-encoder-compatible manager, fallback scoring, thresholds, and tests are implemented. |
| 5. Context assembly | Complete | Format the best chunks into readable LLM context while preserving citation data. | `ContextBuilder`, source tracking, fallback metadata handling, and tests are implemented. |
| 6. Prompt engineering | Complete | Build versioned system and user prompts from context and the query. | `PromptTemplate`, `PromptFactory`, default prompts, example interactions, and focused tests are implemented. |
| 7. LLM integration layer | Complete | Generate an answer through Gemini, with an Ollama fallback and error handling. | Gemini/Claude/Ollama clients, factory selection, retries, normalized responses, and focused tests are implemented. |
| 8. Chat endpoint | Complete | Expose the end-to-end query flow through `/api/chat`. | Answer, sources, metadata, and timing response are implemented and validated. |
| 9. Error handling and resilience | Planned | Keep the pipeline useful when retrieval, ranking, or LLM steps fail. | Retries, fallbacks, graceful degradation, user-friendly errors, and logging remain. |

## Phase 2 Scope Summary

| Area | Status | Notes |
| --- | --- | --- |
| Retrieval pipeline | Tasks 1-4 complete | Vector retrieval, BM25, hybrid merging, and re-ranking are implemented. |
| Context and prompts | Tasks 5-6 complete | Context assembly and versioned prompt templates are implemented. |
| LLM integration | Complete | Gemini client with Claude/Ollama options, fallback, retries, normalized responses, and configuration. |
| Chat endpoint | Planned | `/api/chat` response with sources and timing metadata. |
| Full integration testing | Out of scope | No dedicated end-to-end integration test suite in Phase 2. |

## Phase 1 Update Log

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

## Phase 2 Update Log

| Date | Task | Update |
| --- | --- | --- |
| 2026-09-12 | Planning | Defined Phase 2 scope and removed dedicated integration testing from the Phase 2 plan. |
| 2026-09-12 | Task 1 | Completed LangChain vector retrieval over Weaviate with thresholds, metadata filters, normalized scores, and focused tests. |
| 2026-09-12 | Task 2 | Completed BM25 keyword search with indexing, scoring, normalization, dynamic updates, and focused tests. |
| 2026-09-12 | Task 3 | Completed hybrid retrieval with weighted vector/BM25 merging, deduplication, and focused tests. |
| 2026-09-12 | Task 4 | Completed re-ranking with cross-encoder-compatible scoring, fallback behavior, thresholds, and focused tests. |
| 2026-09-12 | Task 5 | Completed context assembly with source metadata, citation tracking, fallback handling, and focused tests. |
| 2026-09-12 | Task 6 | Completed versioned prompt templates with context/query substitution, optional examples, factory lookup, and focused tests. |
| 2026-09-12 | Task 7 | Completed Claude and Ollama LLM clients, automatic fallback, retries, configuration, documentation, and focused tests. |
| 2026-09-12 | Task 7 follow-up | Added Gemini as the primary configurable provider, with Gemini response parsing, automatic selection, documentation, and tests. |
| 2026-09-12 | Task 8 | Implemented the `/api/chat` endpoint to retrieve context, build prompts, and generate answers with the configured LLM provider. |
