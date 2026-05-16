# UniMind Memory Backend

Simple FastAPI backend for a personalized AI assistant with long-term memory.

The refactored backend uses plain Python logic:

- FastAPI for JSON APIs
- SQLite for durable memory metadata and conversation logs
- FAISS for semantic similarity search
- Gemini embeddings when `GEMINI_API_KEY` is configured
- deterministic local hash embeddings when no key is configured

No LangChain, LangGraph, agents, graph pipelines, ChromaDB, or orchestration frameworks are used.

## Folder Structure

```text
unimind_memory/
  main.py
  config.py
  api/
    chat.py
    memory.py
    routes.py
  memory/
    memory_manager.py
    embedding_service.py
    retrieval.py
    summarizer.py
    storage.py
    fact_extractor.py
  services/
    llm_service.py
  models/
    schemas.py
  storage/
    memory.db
    faiss/
      memory.index
      memory_ids.json
```

## Setup

```bash
cd unimind_memory
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

`GEMINI_API_KEY` is optional. Without it, memory storage and retrieval still work using local deterministic embeddings. Backend `/chat` returns a stable `503` until Gemini is configured.

## APIs

New concise endpoints:

```http
GET /health
POST /chat
POST /memory/store
GET /memory/retrieve?user_id=user123&query=flutter&top_k=5
```

Flutter-compatible aliases are also available under `/api/v1`, including:

```http
GET /api/v1/health
POST /api/v1/memory/exchange
GET /api/v1/memory/context/{user_id}
POST /api/v1/memory/fact
POST /api/v1/memory/search
GET /api/v1/memory/facts/{user_id}
GET /api/v1/memory/episodes/{user_id}
GET /api/v1/memory/short-term/{user_id}
```

Example store request:

```json
{
  "user_id": "user123",
  "content": "User prefers concise explanations",
  "memory_type": "preference",
  "importance": 0.85,
  "source": "manual"
}
```

## Memory Pipeline

1. The user sends a message.
2. The backend stores the message in SQLite.
3. The fact extractor captures personal facts, locations, skills, goals, preferences, and recurring topics.
4. Memory importance is scored by category.
5. Memory text is embedded.
6. SQLite stores durable metadata and the embedding bytes.
7. FAISS stores the vector index for fast similarity search.
8. Lazy decay lowers old low-value memories without deleting them.
9. Retrieval ranks memories with semantic similarity, importance, recency, and keyword overlap.
10. Prompt context is built from relevant facts and episodic summaries within a character budget.

Semantic memory stores durable user facts such as preferences, goals, skills,
locations, and study/work details. Episodic memory stores compact conversation
summaries plus a small recent message slice. These two memory styles support
long-term personalization without overflowing model prompts.

## Semantic Retrieval

Retrieval uses normalized vectors in FAISS `IndexFlatIP`. Final ranking is:

```text
0.55 * semantic_similarity
+ 0.20 * importance
+ 0.15 * recency
+ 0.10 * keyword_overlap
```

Memories with no query evidence receive a small penalty, and duplicate content is
removed. Recency uses a 30-day half-life for ranking. Importance decay is lazy:
old temporary/general memories gradually lose priority, while high-value goals
and personal facts are preserved.

## Debug Mode

Retrieval, search, and context endpoints support optional debug output:

```http
GET /memory/retrieve?user_id=user123&query=python&debug=true
GET /memory/context/user123?query=python&debug=true&max_chars=2000&top_k=5
POST /memory/search
```

When enabled, responses include selected memory ids, semantic scores, keyword
scores, importance scores, recency scores, drop reasons, warnings, and context
budget usage. Default Flutter calls remain lightweight and omit debug details.

## Architecture Notes

See [ARCHITECTURE.md](ARCHITECTURE.md) for an academic-project explanation of
episodic memory, semantic memory, embedding retrieval, contextual prompting,
decay, and Flutter integration.

## Testing

```bash
python -m unittest unimind_memory.smoke_tests
```
