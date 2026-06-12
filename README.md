# OptiRAG — Multi-Modal RAG with Semantic Caching

> A production-ready **Retrieval-Augmented Generation** system that processes text and image documents, routes queries via **RabbitMQ**, caches results semantically in **Redis**, and generates answers locally with **Ollama** — all with **zero OpenAI API costs**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User / API Client                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ POST /api/v1/query
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI  (app/main.py)                          │
│  POST /ingest   POST /query   GET /health   GET /cache/stats        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ submit to local Queue
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│         Local Queue Publisher  (app/queue/local_queue.py)           │
│                                                                     │
│  1. Redis Semantic Cache check ──► HIT: return instantly            │
│                          │                                          │
│                          ▼ MISS                                     │
│  2. ChromaDB top-k retrieval (text + image chunks)                  │
│                          │                                          │
│                          ▼                                          │
│  3. Ollama Generation                                               │
│       ├── llava  (vision model) if images in context                │
│       └── mistral (text model) otherwise                            │
│                          │                                          │
│                          ▼                                          │
│  4. Store result in Redis semantic cache                            │
│                          │                                          │
│                          ▼                                          │
│  5. Return result to API → User                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| API | FastAPI + Uvicorn | REST endpoints |
| Queue | Asyncio Queue | In-process query routing |
| Cache | Redis | Semantic result cache |
| Vector DB | ChromaDB | Embedding storage & retrieval |
| Embeddings | Ollama `nomic-embed-text` | Local text embeddings |
| Text LLM | Ollama `mistral` | Text-only generation |
| Vision LLM | Ollama `llava` | Multimodal generation |
| Config | pydantic-settings | Type-safe .env config |
| Logging | structlog | Structured JSON logs |

---

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Ollama models will be pulled automatically by Docker

### 2. Clone and configure

```bash
git clone <repo-url>
cd opti_rag
cp .env.example .env
```

### 3. Start infrastructure

```bash
docker-compose up -d
```

This starts:
- **Redis** on `localhost:6379`
- **Ollama** on `localhost:11434` (pulls `nomic-embed-text`, `llava`, `mistral`)

> First startup takes several minutes while Ollama downloads the models (~8 GB total).

### 4. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# or: source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: **http://localhost:8000/docs**

---

## Usage Examples

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/report.pdf"
```

```json
{
  "status": "success",
  "filename": "report.pdf",
  "chunks_indexed": 47,
  "message": "Indexed 47 chunks in 3420.1ms."
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the key findings in the executive summary?"}'
```

```json
{
  "answer": "The executive summary highlights three key findings: ...",
  "sources": ["report.pdf"],
  "cache_hit": false,
  "latency_ms": 4821.3,
  "model": "mistral",
  "context_chunks": 5
}
```

### Same query again (semantic cache hit)

```bash
# "Summarise the main findings in the executive summary" → cache hit
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the main findings in the executive summary"}'
```

```json
{
  "answer": "The executive summary highlights three key findings: ...",
  "cache_hit": true,
  "latency_ms": 38.2,
  "similarity": 0.9714,
  "model": "cache"
}
```

> ⚡ **38ms** vs **4821ms** — ~126× faster on semantically similar queries!

### Cache statistics

```bash
curl http://localhost:8000/api/v1/cache/stats
```

```json
{
  "hits": 28,
  "misses": 12,
  "total_lookups": 40,
  "hit_rate": "70.0%",
  "entry_count": 12,
  "threshold": 0.92,
  "ttl_seconds": 86400
}
```

---

## Configuration

All settings are driven by environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TEXT_MODEL` | `mistral` | Text generation model |
| `OLLAMA_VISION_MODEL` | `llava` | Vision+text model |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_CACHE_TTL` | `86400` | Cache entry lifetime (seconds) |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Min cosine similarity for cache hit |
| `CHUNK_SIZE` | `512` | Words per document chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Test coverage:
- `test_cache.py` — Semantic cache logic (miss, hit, flush, stats)
- `test_retrieval.py` — Chunker and ChromaDB vector store
- `test_api.py` — All API endpoints (mocked dependencies)

---

## Project Structure

```
opti_rag/
├── app/
│   ├── api/
│   │   ├── models.py         # Pydantic request/response schemas
│   │   └── routes.py         # FastAPI endpoints
│   ├── cache/
│   │   └── semantic_cache.py # Redis semantic cache (cosine similarity)
│   ├── core/
│   │   ├── config.py         # Settings (pydantic-settings)
│   │   └── logging.py        # Structured logging (structlog)
│   ├── generation/
│   │   └── llm_client.py     # Ollama LLM client (mistral + llava)
│   ├── ingestion/
│   │   ├── chunker.py        # Sliding-window text chunker
│   │   ├── document_loader.py# PDF / image / text / DOCX loaders
│   │   └── embedder.py       # Ollama embedding client
│   ├── queue/
│   │   └── local_queue.py    # Local in-process queue & RAG worker
│   ├── retrieval/
│   │   ├── retriever.py      # Query embedding + top-k retrieval
│   │   └── vector_store.py   # ChromaDB wrapper
│   └── main.py               # FastAPI app + lifespan
├── data/
│   ├── uploads/              # Uploaded documents
│   └── chroma_db/            # Persisted ChromaDB index
├── tests/
│   ├── test_api.py
│   ├── test_cache.py
│   └── test_retrieval.py
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Semantic Cache — How It Works

```
Query: "What is the revenue forecast for 2025?"
              │
              ▼
        Embed with Ollama
              │
              ▼
    For each cached entry:
        cosine_similarity(query_vec, cached_vec)
              │
        ┌─────┴──────┐
        │ sim ≥ 0.92 │  → Cache HIT → return stored answer
        └─────┬──────┘
              │ sim < 0.92
              ▼
         Run full RAG pipeline
              │
              ▼
    Store {query, answer, embedding} in Redis (TTL 24h)
```

The threshold of **0.92** is tunable via `SEMANTIC_CACHE_THRESHOLD` in `.env`.
Raising it makes the cache more conservative (fewer hits); lowering it
makes it more aggressive (may return answers to loosely related questions).

---

## Monitoring

- **API Docs**: http://localhost:8000/docs
- **Cache Stats**: `GET /api/v1/cache/stats`
- **Health**: `GET /api/v1/health`
