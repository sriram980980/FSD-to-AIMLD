# Dependencies: langchain>=0.2, sentence-transformers>=2.7, chromadb>=0.4,
#               fastapi>=0.111, uvicorn>=0.29, numpy>=1.24
# Node: 4.3 — Augmented Generation — RAG & Vector Search
# Run (standalone test): python starter.py
# Run (API server):       uvicorn starter:app --reload

from __future__ import annotations

import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG QA API", version="1.0")
app.state.collection: Optional[chromadb.Collection] = None


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    documents: list[str]


class IngestResponse(BaseModel):
    status: str
    documents_received: int
    chunks_indexed: int


class AskRequest(BaseModel):
    query: str
    top_k: int = 3


class AskResponse(BaseModel):
    query: str
    top_k_chunks: list[dict]
    rag_prompt: str


# ---------------------------------------------------------------------------
# Implemented helper 1 — sample corpus (reduces boilerplate for student)
# ---------------------------------------------------------------------------
def load_sample_corpus() -> list[str]:
    """Return 8 FSD-themed documents that serve as the RAG knowledge base."""
    return [
        (
            "FastAPI is a modern Python web framework built on Starlette and Pydantic. "
            "It auto-generates OpenAPI docs and supports async request handlers natively. "
            "Declare endpoints with type-annotated function signatures; FastAPI validates "
            "all incoming JSON automatically using Pydantic models."
        ),
        (
            "Uvicorn is an ASGI server implementation for Python. It uses uvloop and "
            "httptools for high-performance HTTP serving. Start a FastAPI app with: "
            "uvicorn app:app --reload. For production, run behind Nginx as a reverse proxy."
        ),
        (
            "ChromaDB is an open-source vector database optimized for storing embeddings. "
            "It supports in-memory and persistent modes and performs approximate nearest-"
            "neighbour search using HNSW. Collections store (id, embedding, document, "
            "metadata) tuples and expose a query() interface for top-k similarity search."
        ),
        (
            "Sentence Transformers produce dense vector embeddings from text. The model "
            "all-MiniLM-L6-v2 outputs 384-dimensional float32 vectors and runs on CPU "
            "with acceptable latency. It is trained via contrastive learning on sentence "
            "pairs so semantically similar sentences cluster together in vector space."
        ),
        (
            "RAG (Retrieval-Augmented Generation) pairs a retriever with a language model. "
            "The retriever fetches relevant chunks from a vector store; the language model "
            "conditions its generation on those chunks. The model never sees the full "
            "corpus — only the top-k most relevant passages fit inside the context window."
        ),
        (
            "Token limits constrain how much context an LLM can process at once. GPT-4 "
            "supports 128 k tokens; open-weight models like Llama-3-8B typically cap at "
            "8 k tokens. Chunking strategy must keep each chunk well under this limit so "
            "multiple chunks fit alongside the question and system prompt."
        ),
        (
            "Cosine similarity measures the angle between two vectors regardless of their "
            "magnitude. Values range from -1 (opposite direction) to +1 (identical "
            "direction). In retrieval, chunks whose embedding points in the same direction "
            "as the query embedding are ranked highest by the vector store."
        ),
        (
            "LangChain provides composable primitives for building LLM applications. Its "
            "LCEL (LangChain Expression Language) lets you chain retrievers, prompts, and "
            "models with the pipe operator. LangChain4j mirrors these abstractions for JVM "
            "services, making RAG pipelines portable across Python and Java backends."
        ),
    ]


# ---------------------------------------------------------------------------
# Implemented helper 2 — manual cosine similarity for verification
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D NumPy arrays."""
    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Implemented helper 3 — pretty-print retrieval results
# ---------------------------------------------------------------------------
def format_retrieval_results(results: list[dict]) -> str:
    """Return a formatted string of retrieval hits for console display."""
    lines: list[str] = []
    for rank, hit in enumerate(results, start=1):
        preview = hit["text"][:72].replace("\n", " ")
        lines.append(f"  [{rank}] sim={hit['similarity']:.4f} | {preview}...")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 1 — student implements this
# ---------------------------------------------------------------------------
def chunk_documents(
    texts: list[str],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[tuple[str, str]]:
    """Split each text into overlapping character windows.

    Returns a list of (chunk_id, chunk_text) tuples.
    chunk_id format: "doc{doc_index}_chunk{chunk_index}"

    Steps:
    - For each text, slide a window of `chunk_size` characters.
    - Advance by (chunk_size - overlap) on each step so adjacent chunks share
      `overlap` characters.
    - Strip whitespace from each chunk and skip empty ones.
    - Assign a unique chunk_id per chunk.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 2 — student implements this
# ---------------------------------------------------------------------------
def build_collection(chunks: list[tuple[str, str]]) -> chromadb.Collection:
    """Embed all chunks and store them in an ephemeral ChromaDB collection.

    Steps:
    - Create a chromadb.Client() (ephemeral, in-memory).
    - Create a collection named "rag_qa" with:
        embedding_function = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        metadata = {"hnsw:space": "cosine"}
    - Call collection.add(documents=[...], ids=[...]) with all chunks in one call.
    - Return the populated collection.

    Hint: unzip the list of tuples into two parallel lists (ids, documents).
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 3 — student implements this
# ---------------------------------------------------------------------------
def retrieve_top_k(
    collection: chromadb.Collection,
    query: str,
    k: int = 3,
) -> list[dict]:
    """Return the top-k most similar chunks for a query.

    Steps:
    - Call collection.query(query_texts=[query], n_results=k,
                            include=["documents", "distances"]).
    - ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite).
      Convert to similarity: similarity = 1.0 - distance.
    - Return a list of dicts sorted by similarity descending:
        [{"text": str, "similarity": float}, ...]
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 4 — student implements this
# ---------------------------------------------------------------------------
def assemble_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build the final RAG prompt from query and retrieved chunks.

    Required structure (exact):

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

    Each [Context N] block is separated by a blank line.
    The prompt must end with "Answer:" (no trailing newline required).
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 5 — POST /ingest endpoint (replace the raise with your implementation)
# ---------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Index the submitted documents into ChromaDB.

    Steps:
    1. Call chunk_documents(request.documents, chunk_size=400, overlap=80).
    2. Call build_collection(chunks) and store the result in app.state.collection.
    3. Return IngestResponse with status="ok", the document count, and chunk count.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 6 — POST /ask endpoint (replace the raise with your implementation)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Liveness check — always returns 200."""
    indexed = app.state.collection is not None
    return {"status": "ok", "index_ready": indexed}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a question using the indexed document collection.

    Steps:
    1. If app.state.collection is None, raise HTTPException(status_code=400,
       detail="index not built — call /ingest first").
    2. Call retrieve_top_k(app.state.collection, request.query, k=request.top_k).
    3. Call assemble_rag_prompt(request.query, chunks).
    4. Return AskResponse with query, top_k_chunks, and rag_prompt.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Standalone pipeline test — runs Tasks 1-4 without the HTTP server
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== RAG QA API — Standalone Pipeline Test ===\n")

    # Implemented helper — always works
    corpus = load_sample_corpus()
    print(f"[Setup] Loaded {len(corpus)} documents")

    # Implemented helper — verify cosine math before touching ChromaDB
    q_vec = np.array([0.6, 0.8])
    c_vec = np.array([0.9, 0.4])
    sim = cosine_similarity(q_vec, c_vec)
    print(f"[Verify] cosine_similarity([0.6, 0.8], [0.9, 0.4]) = {sim:.4f}  (expected ~0.873)\n")

    # Task 1
    try:
        chunks = chunk_documents(corpus, chunk_size=400, overlap=80)
        print(f"[Task 1] chunk_documents → {len(chunks)} chunks  (expect: 10–16)")
    except NotImplementedError as exc:
        print(f"[Task 1] {exc}")
        print("\n[API] Start the server: uvicorn starter:app --reload")
        return

    # Task 2
    try:
        collection = build_collection(chunks)
        print(f"[Task 2] build_collection → collection with {collection.count()} items")
    except NotImplementedError as exc:
        print(f"[Task 2] {exc}")
        return

    # Task 3
    try:
        query = "How does FastAPI handle async requests?"
        results = retrieve_top_k(collection, query, k=3)
        print(f"[Task 3] retrieve_top_k query: {query!r}")
        print(format_retrieval_results(results))
    except NotImplementedError as exc:
        print(f"[Task 3] {exc}")
        return

    # Task 4
    try:
        prompt = assemble_rag_prompt(query, results)
        print(f"\n[Task 4] assemble_rag_prompt → {len(prompt)} char prompt")
        preview = prompt[:400] + "..." if len(prompt) > 400 else prompt
        print("\n--- Prompt Preview (first 400 chars) ---")
        print(preview)
    except NotImplementedError as exc:
        print(f"[Task 4] {exc}")
        return

    print("\n[API] Start the server: uvicorn starter:app --reload")
    print("[API] POST /ingest  — index your document corpus")
    print("[API] POST /ask     — query the indexed documents")


if __name__ == "__main__":
    main()
