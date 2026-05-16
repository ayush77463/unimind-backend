# UniMind Memory Architecture

## Overview

UniMind uses a simple long-term memory pipeline for personalization:

```text
Flutter chat -> FastAPI -> extraction -> importance scoring -> SQLite
                                      -> embeddings -> FAISS retrieval
                                      -> compact context -> model prompt
```

The design intentionally avoids orchestration frameworks. Each step is plain
Python so the system is easy to explain, test, and maintain for an academic
project.

## Memory Types

### Semantic Memory

Semantic memory stores durable facts about the user. Examples include:

- "User prefers concise Python explanations"
- "User wants to become an AI engineer"
- "User is currently in Patna"

The public memory types stay simple: `semantic`, `preference`, and `episodic`.
More specific labels such as `goal`, `skill`, `location`, or `study` are stored
in metadata as `category`.

### Episodic Memory

Episodic memory stores a compact summary of a past conversation. It does not
store the entire conversation as prompt context. The backend saves a short
summary plus a small recent message slice so the memory viewer can inspect what
happened without overloading future prompts.

## Extraction And Scoring

The deterministic extractor works without an API key. It detects:

- preferences and dislikes
- goals
- names and identity facts
- location and hometown facts
- study, work, and skill facts
- recurring topics from repeated questions

Optional Gemini extraction can add extra memories, but failures fall back to the
local extractor.

Importance is assigned by category:

```text
goals / personal facts      high
location / study / work     medium-high
preferences / skills        medium
recurring topics            medium once repeated
general or temporary chat   low
```

## Decay

Memory decay is lazy. It runs during retrieval and status/viewer calls instead
of using a background scheduler.

Old low-value memories gradually lose importance. High-value goals, identity,
and personal facts are preserved with a strong floor. Decay never deletes user
memories automatically; it only changes ranking priority.

## Embedding Retrieval

Each memory is embedded and stored in SQLite. FAISS stores only the vector index
and can be rebuilt from SQLite at any time.

Retrieval uses hybrid ranking:

```text
0.55 * semantic similarity
+ 0.20 * importance
+ 0.15 * recency
+ 0.10 * keyword overlap
```

Memories with no query evidence receive a penalty. Duplicate content is removed
from results. Debug mode explains selected and dropped memories.

## Contextual Prompting

The context builder retrieves relevant semantic and episodic memories, then
fits them into a lightweight character budget. Important and relevant facts are
kept first. This prevents token overflow and keeps Flutter responses fast.

The model prompt receives a compact section called `MEMORY CONTEXT`. The user
does not see internal scoring or memory mechanics unless debug mode is enabled.

## API And Flutter Integration

Flutter continues to call the same endpoints:

- `GET /api/v1/memory/context/{user_id}`
- `POST /api/v1/memory/exchange`
- `POST /api/v1/memory/search`
- viewer endpoints for facts, episodes, and short-term messages

Responses keep stable fields such as `success`, `context`, `facts`, `episodes`,
`results`, and counts. Optional `warnings` and `debug` fields are added only for
diagnostics.

## Failure Handling

The memory backend is designed to degrade gracefully:

- Gemini embedding failures fall back to local hash embeddings.
- LLM extraction and summarization timeouts use deterministic local logic.
- FAISS load/search failures trigger one rebuild from SQLite.
- API responses remain predictable so Flutter can continue chatting even when
  the memory service is offline or partially degraded.
