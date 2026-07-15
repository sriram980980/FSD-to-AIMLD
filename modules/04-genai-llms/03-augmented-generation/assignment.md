# Assignment 4.3 — Build Full-Stack RAG Document QA API

## Objective

Build a FastAPI service that ingests plain-text documents, indexes them in ChromaDB using sentence-transformer embeddings, and answers natural-language questions via semantic retrieval — proving you can wire a complete three-stage RAG pipeline into a deployable HTTP API.

## Background

The lesson showed that RAG separates knowledge storage (the vector index) from the language model. New documents enter the system by re-indexing, not by retraining. This assignment makes that pipeline concrete: you implement every stage — chunking, embedding, retrieval, and prompt assembly — then expose it behind two REST endpoints so any LLM frontend can call it. Review the three-stage diagram in `lesson.md` (Indexing → Retrieval → Generation) before starting; the six tasks below map directly onto those stages plus the API layer.

## Setup

```bash
pip install langchain>=0.2 sentence-transformers>=2.7 chromadb>=0.4 fastapi>=0.111 uvicorn>=0.29 numpy>=1.24
```

> **CPU note:** `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) on first run. Subsequent runs use the local cache. No GPU required.

## Tasks

### Task 1 — Implement `chunk_documents()`

Open `starter/starter.py`. Implement `chunk_documents(texts, chunk_size, overlap)` so it:

- Iterates over every string in `texts`.
- Splits each string into overlapping windows of at most `chunk_size` characters.
- Advances the window by `chunk_size - overlap` characters on each step so adjacent chunks share `overlap` characters.
- Returns a list of `(chunk_id, chunk_text)` tuples where `chunk_id` is a unique string such as `"doc0_chunk2"`.

Run `python starter/starter.py` — the `[Task 1]` line must print a chunk count **≥ len(texts)**.

### Task 2 — Implement `build_collection()`

Implement `build_collection(chunks)` so it:

- Creates an **ephemeral** ChromaDB client (`chromadb.Client()`).
- Creates a collection named `"rag_qa"` with `SentenceTransformerEmbeddingFunction` using `all-MiniLM-L6-v2` and cosine space.
- Adds all chunks in a single `collection.add()` call, passing the texts as `documents` and the chunk IDs as `ids`.
- Returns the populated `chromadb.Collection`.

Run `python starter/starter.py` — the `[Task 2]` line must print `collection with N items` where N matches the Task 1 chunk count.

### Task 3 — Implement `retrieve_top_k()`

Implement `retrieve_top_k(collection, query, k)` so it:

- Calls `collection.query(query_texts=[query], n_results=k, include=["documents","distances"])`.
- Converts ChromaDB's cosine **distance** to cosine **similarity** via `similarity = 1.0 - distance`.
- Returns a list of `k` dicts, each with keys `"text"` (str) and `"similarity"` (float), sorted by similarity descending.

Run `python starter/starter.py` — the `[Task 3]` block must list 3 results with similarity scores. The top result for the default query should have `similarity > 0.60`.

### Task 4 — Implement `assemble_rag_prompt()`

Implement `assemble_rag_prompt(query, context_chunks)` so it builds a prompt string with this exact structure:

```
You are a technical documentation assistant.
Answer the question using ONLY the provided context.
If the context does not contain the answer, reply: 'I don't have that information.'

[Context 1]
<chunk text>

[Context 2]
<chunk text>

...

Question: <query>

Answer:
```

Run `python starter/starter.py` — the `[Task 4]` line must print the assembled prompt length and a preview.

### Task 5 — Wire the `POST /ingest` endpoint

In `starter/starter.py`, find the `ingest()` route handler (already stubbed). Replace the `raise NotImplementedError` with logic that:

1. Calls `chunk_documents()` on `request.documents` with `chunk_size=400` and `overlap=80`.
2. Calls `build_collection()` on the resulting chunks and stores the result in `app.state.collection`.
3. Returns `{"status": "ok", "documents_received": N, "chunks_indexed": M}`.

Test with:

```bash
uvicorn starter.starter:app --reload
# In a second terminal:
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["FastAPI supports async handlers.", "ChromaDB stores embeddings."]}' | python -m json.tool
```

Expected: `chunks_indexed` ≥ 2 and `status` is `"ok"`.

### Task 6 — Wire the `POST /ask` endpoint

Find the `ask()` route handler (already stubbed). Replace the `raise NotImplementedError` with logic that:

1. Returns `{"error": "index not built — call /ingest first"}` with HTTP 400 if `app.state.collection` is `None`.
2. Calls `retrieve_top_k()` with `k=request.top_k`.
3. Calls `assemble_rag_prompt()` on the query and retrieved chunks.
4. Returns `{"query": ..., "top_k_chunks": [...], "rag_prompt": ...}`.

Test with:

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ChromaDB used for?", "top_k": 2}' | python -m json.tool
```

Expected: `top_k_chunks` contains 2 entries, each with `text` and `similarity` keys. `rag_prompt` starts with `"You are a technical documentation assistant."`.

## Expected Output

Running `python starter/starter.py` after completing Tasks 1–4 produces output in this shape:

```
=== RAG QA API — Standalone Pipeline Test ===

[Setup] Loaded 8 documents
[Verify] cosine_similarity([0.6, 0.8], [0.9, 0.4]) = 0.8727  (expected ~0.873)

[Task 1] chunk_documents → 11 chunks  (expect: 10–16)
[Task 2] build_collection → collection with 11 items
[Task 3] retrieve_top_k query: 'How does FastAPI handle async requests?'
  [1] sim=0.7843 | FastAPI is a modern Python web framework. It supports async...
  [2] sim=0.5512 | Uvicorn is an ASGI server for Python. Start with: uvicorn ...
  [3] sim=0.4109 | RAG combines a retriever with a generator. The retriever fe...
[Task 4] assemble_rag_prompt → 631 char prompt

--- Prompt Preview (first 400 chars) ---
You are a technical documentation assistant.
Answer the question using ONLY the provided context.
If the context does not contain the answer, reply: 'I don't have that information.'

[Context 1]
FastAPI is a modern Python web framework...

[API] Start the server: uvicorn starter.starter:app --reload
[API] POST /ingest  — index your document corpus
[API] POST /ask     — query the indexed documents
```

> Similarity scores vary ±0.05 between runs depending on model version. Chunk count varies with corpus length.

## Evaluation Criteria

- [ ] `chunk_documents()` produces overlapping chunks — adjacent chunks share at least `overlap` characters.
- [ ] `build_collection()` inserts all chunks into ChromaDB and returns a collection where `collection.count()` equals the number of chunks.
- [ ] `retrieve_top_k()` returns results sorted by similarity descending, with `similarity = 1.0 - distance`.
- [ ] `assemble_rag_prompt()` contains all context chunks labeled `[Context N]` and ends with `"Answer:"`.
- [ ] `POST /ingest` returns HTTP 200 with `chunks_indexed > 0` after receiving a list of documents.
- [ ] `POST /ask` returns HTTP 400 when called before `/ingest`, and HTTP 200 with `rag_prompt` and `top_k_chunks` after indexing.
- [ ] `python starter/starter.py` runs end-to-end without uncaught exceptions after Tasks 1–4 are implemented.

## Extension Challenge

Implement **metadata-filtered retrieval**: when a document is ingested, attach a `source` metadata field (e.g., filename or URL). Extend the `POST /ask` endpoint to accept an optional `source_filter` string that restricts retrieval to chunks from that source using ChromaDB's `where` clause. Then measure and print the hit-rate difference between filtered and unfiltered retrieval across five queries against a mixed corpus of at least three distinct sources. No starter code is provided for this extension.
