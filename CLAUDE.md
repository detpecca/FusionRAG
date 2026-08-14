# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FusionRAG is a GraphRAG knowledge-base QA system: on top of vector-retrieval RAG it builds an entity/relationship knowledge graph and answers via three-way hybrid retrieval: `local` (entity side) + `global` (relation side) + `naive` (chunk side), with optional rerank and answer references. Zero external DB dependency — all state is JSON/numpy/NetworkX files under `WORKING_DIR` (KV optionally SQLite). Extensive design docs live in `docs/` and `README.md` (Chinese).

## Commands

```bash
# Setup (Windows Git Bash)
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env          # must set LLM_API_KEY; any OpenAI-compatible endpoint works

# Run API server -> http://localhost:8000 (Swagger at /docs, web UI at /)
python -m fusionrag

# Tests (deterministic Fake LLM/Embedding, no network needed)
pip install -r requirements-dev.txt
pytest -q
pytest tests/test_pipeline.py -q            # single file
pytest tests/test_merge.py::test_name -q    # single test

# Docker
docker compose up -d --build

# Real-endpoint smoke scripts (need a live .env)
python scripts/e2e_real.py
python scripts/verify_delete.py
```

`pytest.ini` sets `asyncio_mode = auto`, so `async def test_*` run without decorators.

## Architecture

`FusionRAG` (`fusionrag/core.py`) is the central class wiring two async pipelines over shared storage. Config is a single dataclass `FusionRAGConfig` (`config.py`), loaded from env via `.from_env()`.

**Insert pipeline** (`ainsert`, serialized by `self._insert_lock`):
token-window chunking (`chunking.py`) → chunks to KV + vector store → LLM entity/relation extraction with one gleaning pass (`extract.py`) → dedup/merge into graph + entity/relation vector stores (`merge.py`) → status + flush.

**Query pipeline** (`aquery` → `rag_query` in `query.py`):
LLM extracts high/low-level keywords → `local`/`global`/`naive` retrieval → round-robin merge → optional rerank (`rerank.py`, graceful fallback to original order on failure) → token-budget truncation → assemble KG context → LLM generation (streaming/history-aware) → reference list (`references`: chunk → doc title/excerpt) attached to `raw_data`.

**Module map**:
- `core.py` — `FusionRAG` orchestrator; also `adelete_by_doc_id` (ledger-based deletion consistency + survivor description rebuild via LLM-cache re-extraction, `DELETE_REBUILD_DESCRIPTIONS`)
- `chunking.py` — token sliding-window split (`tiktoken`, char-level fallback offline)
- `extract.py` — delimiter-format entity/relation extraction + gleaning
- `merge.py` — dedup, `<SEP>`-joined descriptions, LLM summary on overflow, batch vector upsert
- `query.py` — `QueryParam`/`QueryResult`, keyword extraction, 3-way retrieval, context assembly, generation
- `rerank.py` — Jina/Cohere-style rerank client (`POST {base_url}/rerank`), low-score filter + top_n; no-op unless `RERANK_MODEL` set
- `storage.py` — `JsonKVStorage`, `SimpleVectorStorage` (numpy), `NetworkXGraphStorage`
- `llm.py` — `LLMService`: OpenAI-compatible LLM/embedding client, KV result cache, exponential-backoff retry on 429/5xx
- `sessions.py` — `SessionStore`, sliding-window chat history (history feeds generation, NOT retrieval)
- `prompts.py` — extraction/query prompt templates, `GRAPH_FIELD_SEP`
- `api.py` — FastAPI app (REST + SSE), serves `webui/index.html`
- `__main__.py` — uvicorn entrypoint for `python -m fusionrag`

## Key conventions

- **Storage is file-backed**: writes buffer in memory until `index_done_callback()`; `core._flush()` fans out to all stores. Deterministic IDs via `compute_mdhash_id` (`utils.py`) — chunk keys are recomputable from `doc_id`+index, so no chunk list is persisted.
- **Idempotency**: re-importing identical content is skipped (`PROCESSED` check); deletion is a no-op if already applied.
- **Relations are undirected**: keyed by sorted `(src, tgt)` joined with `GRAPH_FIELD_SEP`; both `rel-<src+tgt>` and `rel-<tgt+src>` vector ids are maintained.
- **Changing the embedding model invalidates stored vectors** — startup refuses to boot on model/dim mismatch (fingerprint in `embedding_meta.json`); clear `WORKING_DIR` and re-import (`llm_cache.json` can be kept).
- **Tests inject Fake LLM/Embedding/Rerank** via `tests/conftest.py` and the `llm_func`/`embedding_func`/`rerank_func` config hooks — no external API in the suite.
- Docs/README/prompts are Chinese; `SUMMARY_LANGUAGE`/`language` config controls extraction+answer language.
