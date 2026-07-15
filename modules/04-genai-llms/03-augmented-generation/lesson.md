# 4.3 — Augmented Generation — RAG & Vector Search

## Hook

RAG is an API gateway pattern — instead of the LLM hallucinating from memory, it routes each query to a vector DB (a specialized microservice), retrieves the most relevant document chunks, and injects them into the prompt the same way an API gateway forwards request context to the right backend service.

## The Problem

LLMs are frozen at their training cutoff. Ask a vanilla GPT about your internal wiki, last quarter's architecture decision records, or a document uploaded five minutes ago — it either hallucinates plausible-sounding nonsense or admits ignorance. Fine-tuning the whole model every time new documents arrive is prohibitively slow and expensive. RAG solves this by keeping the model weights static and instead building a live, queryable index of your documents; the model gets accurate context at inference time, not baked-in memorization.

## Theory

### Embeddings and Cosine Similarity

An embedding function $E$ maps any text string to a fixed-size dense vector:

$$\mathbf{v} = E(t) \in \mathbb{R}^d$$

where:
- $t$ = input text (a sequence of tokens)
- $\mathbf{v}$ = resulting embedding vector
- $d$ = embedding dimension (e.g., $d = 384$ for `all-MiniLM-L6-v2`)

To rank which stored chunks best answer a query, compute **cosine similarity** between the query embedding $\mathbf{q}$ and each chunk embedding $\mathbf{c}$:

$$\text{sim}(\mathbf{q}, \mathbf{c}) = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\|\,\|\mathbf{c}\|}$$

where:
- $\mathbf{q} \cdot \mathbf{c}$ = dot product — sum of element-wise products
- $\|\mathbf{q}\|$ = L2 norm of the query vector $= \sqrt{\sum_i q_i^2}$
- Result ranges from $-1$ (opposite) to $+1$ (identical direction)

**Worked numeric example** (2D vectors for clarity):

| Symbol | Value |
|--------|-------|
| $\mathbf{q}$ | $[0.6,\ 0.8]$ |
| $\mathbf{c}_A$ | $[0.9,\ 0.4]$ |
| $\mathbf{c}_B$ | $[-0.7,\ 0.3]$ |

Compute norms:

$$\|\mathbf{q}\| = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = 1.0$$

$$\|\mathbf{c}_A\| = \sqrt{0.9^2 + 0.4^2} = \sqrt{0.81 + 0.16} \approx 0.985$$

$$\|\mathbf{c}_B\| = \sqrt{(-0.7)^2 + 0.3^2} = \sqrt{0.49 + 0.09} \approx 0.762$$

Compute similarities:

$$\text{sim}(\mathbf{q}, \mathbf{c}_A) = \frac{(0.6)(0.9) + (0.8)(0.4)}{1.0 \times 0.985} = \frac{0.54 + 0.32}{0.985} \approx \mathbf{0.873}$$

$$\text{sim}(\mathbf{q}, \mathbf{c}_B) = \frac{(0.6)(-0.7) + (0.8)(0.3)}{1.0 \times 0.762} = \frac{-0.42 + 0.24}{0.762} \approx \mathbf{-0.236}$$

Chunk A is semantically close to the query; Chunk B points in the opposite direction. The retriever returns Chunk A as the top result.

### Chunking Strategy

Raw documents must be split into chunks before embedding. A naive whitespace split breaks mid-sentence, severing the context the model needs. The standard approach is **recursive character splitting with overlap**:

- `chunk_size` — maximum characters per chunk (e.g., 512)
- `chunk_overlap` — characters shared between adjacent chunks (e.g., 64)

Overlap prevents the retriever from missing information that straddles a boundary — the same principle as overlapping log-rotation windows for atomic consistency.

### The Three-Stage RAG Pipeline

```
Stage 1 — Indexing  (offline, run once or incrementally)
  Document → chunk() → embed() → store(text, vector) in vector DB

Stage 2 — Retrieval  (online, per query)
  Query → embed() → top-k similarity search → ranked chunks

Stage 3 — Generation  (online, per query)
  system_prompt + retrieved_chunks + user_query → LLM → answer
```

The LLM never sees your raw documents — it sees only the top-k chunks assembled into a prompt. This keeps latency low and context windows focused.

## Python Implementation

```python
# Dependencies: sentence-transformers>=2.7, chromadb>=0.4, numpy>=1.24

import textwrap
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 1. Sample knowledge base — simulates private documentation
# ---------------------------------------------------------------------------
DOCUMENTS: List[str] = [
    "FastAPI is a modern Python web framework built on Starlette and Pydantic. "
    "It auto-generates OpenAPI docs and supports async request handlers natively.",
    "Uvicorn is an ASGI server implementation for Python. It uses uvloop and httptools "
    "for high-performance HTTP serving. Start with: uvicorn app:app --reload",
    "ChromaDB is an open-source vector database optimized for storing embeddings. "
    "It supports in-memory and persistent modes with cosine similarity search.",
    "Sentence Transformers produce dense vector embeddings from text. "
    "The all-MiniLM-L6-v2 model outputs 384-dimensional vectors and runs on CPU.",
    "RAG (Retrieval-Augmented Generation) pairs a retriever with a generator. "
    "The retriever fetches relevant chunks; the generator conditions on them to answer.",
    "Token limits restrict how much context an LLM can process at once. "
    "GPT-4 supports 128k tokens; smaller models cap at 4k or 8k tokens.",
]


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def chunk_document(text: str, chunk_size: int = 300, overlap: int = 60) -> List[str]:
    """Split text into overlapping chunks of at most chunk_size characters."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# 3. Indexing — embed + store
# ---------------------------------------------------------------------------
def build_vector_store(documents: List[str]) -> chromadb.Collection:
    """Embed all document chunks and load them into an in-memory ChromaDB collection."""
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.Client()  # ephemeral in-memory client
    collection = client.create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: List[str] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(documents=all_chunks, ids=ids)

    print(f"[Index] {len(documents)} documents → {len(all_chunks)} chunks stored")
    return collection


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    collection: chromadb.Collection, query: str, top_k: int = 3
) -> List[str]:
    """Return top_k chunks most similar to the query."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances"],
    )
    chunks: List[str] = results["documents"][0]
    distances: List[float] = results["distances"][0]

    print(f"\n[Retrieve] Query: {query!r}")
    print(f"[Retrieve] Top {top_k} results (cosine similarity):")
    for rank, (chunk, dist) in enumerate(zip(chunks, distances), start=1):
        similarity = 1.0 - dist  # ChromaDB stores cosine distance = 1 - similarity
        print(f"  [{rank}] sim={similarity:.4f} | {chunk[:72]}...")
    return chunks


# ---------------------------------------------------------------------------
# 5. Prompt assembly
# ---------------------------------------------------------------------------
def build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """Assemble the final RAG prompt ready to send to any LLM."""
    context_block = "\n\n".join(
        f"[Context {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        "You are a technical documentation assistant.\n"
        "Answer the question using ONLY the provided context.\n"
        "If the context does not contain the answer, reply: "
        "'I don't have that information.'\n\n"
        f"{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# 6. Manual cosine similarity verification
# ---------------------------------------------------------------------------
def verify_cosine_similarity(query: str, document: str) -> None:
    """Compute cosine similarity manually to cross-check ChromaDB results."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec: np.ndarray = model.encode(query)
    d_vec: np.ndarray = model.encode(document)
    cosine_sim: float = float(
        np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
    )
    print(f"\n[Verify] Manual cosine_sim(query, doc[3]) = {cosine_sim:.4f}")
    print(f"[Verify] Embedding dimension: {q_vec.shape[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== RAG Pipeline: Indexing → Retrieval → Prompt Assembly ===\n")

    # Stage 1: Indexing
    collection = build_vector_store(DOCUMENTS)

    # Stage 2: Retrieval
    query = "What embedding model runs on CPU and what size vectors does it produce?"
    retrieved_chunks = retrieve(collection, query, top_k=3)

    # Stage 3: Prompt assembly (swap in your LLM call here)
    prompt = build_rag_prompt(query, retrieved_chunks)
    print("\n[Prompt] Assembled RAG prompt:")
    print(textwrap.indent(prompt, prefix="  "))

    # Manual verification
    verify_cosine_similarity(query, DOCUMENTS[3])


if __name__ == "__main__":
    main()
```

**What to notice in the output:**

- The `[Index]` line confirms chunk count — short documents may produce only one chunk each, longer documents produce several.
- The similarity scores in `[Retrieve]` should place the Sentence Transformers document (`DOCUMENTS[3]`) at rank 1 with `sim > 0.75` — it is the document that directly answers the query.
- The `[Prompt]` block shows exactly what the LLM receives: your system instruction, the ranked context chunks, and the user question. Nothing from the other five documents leaks in.
- The `[Verify]` line manually recomputes cosine similarity with NumPy — it should match the ChromaDB score within ±0.01, confirming the vector store is doing exactly what the math describes.

**Sample output:**

```
=== RAG Pipeline: Indexing → Retrieval → Prompt Assembly ===

[Index] 6 documents → 6 chunks stored

[Retrieve] Query: 'What embedding model runs on CPU and what size vectors does it produce?'
[Retrieve] Top 3 results (cosine similarity):
  [1] sim=0.8312 | Sentence Transformers produce dense vector embeddings from text. The all-Min...
  [2] sim=0.5147 | RAG (Retrieval-Augmented Generation) pairs a retriever with a generator. The...
  [3] sim=0.4081 | ChromaDB is an open-source vector database optimized for storing embeddings....

[Prompt] Assembled RAG prompt:
  You are a technical documentation assistant.
  Answer the question using ONLY the provided context.
  ...
  [Context 1]
  Sentence Transformers produce dense vector embeddings from text.
  The all-MiniLM-L6-v2 model outputs 384-dimensional vectors and runs on CPU.
  ...

[Verify] Manual cosine_sim(query, doc[3]) = 0.8309
[Verify] Embedding dimension: 384
```

## Java Implementation

```java
// Dependencies (Maven):
// dev.langchain4j:langchain4j:0.31.0
// dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2-q:0.31.0

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class RagPipelineDemo {

    // ---------------------------------------------------------------------------
    // Sample knowledge base
    // ---------------------------------------------------------------------------
    private static final List<String> DOCUMENTS = List.of(
        "FastAPI is a modern Python web framework built on Starlette and Pydantic. " +
        "It auto-generates OpenAPI docs and supports async request handlers natively.",

        "Uvicorn is an ASGI server for Python using uvloop for high-performance HTTP. " +
        "Start with: uvicorn app:app --reload",

        "ChromaDB is an open-source vector database for storing embeddings. " +
        "It supports in-memory and persistent modes with cosine similarity search.",

        "Sentence Transformers produce dense vector embeddings from text. " +
        "The all-MiniLM-L6-v2 model outputs 384-dimensional vectors and runs on CPU.",

        "RAG combines a retriever with a generator. The retriever fetches relevant " +
        "chunks; the generator conditions on them to produce accurate answers.",

        "Token limits restrict how much context an LLM can process. " +
        "GPT-4 supports 128k tokens; smaller models cap at 4k or 8k tokens."
    );

    // ---------------------------------------------------------------------------
    // Stage 1: Index documents
    // ---------------------------------------------------------------------------
    static EmbeddingStore<TextSegment> buildVectorStore(
            EmbeddingModel embeddingModel,
            List<String> rawDocs) {

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 60);

        int totalChunks = 0;
        for (String rawDoc : rawDocs) {
            Document document = Document.from(rawDoc);
            List<TextSegment> segments = splitter.splitDocument(document);
            List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
            store.addAll(embeddings, segments);
            totalChunks += segments.size();
        }

        System.out.printf("[Index] %d documents → %d chunks stored%n",
                rawDocs.size(), totalChunks);
        return store;
    }

    // ---------------------------------------------------------------------------
    // Stage 2: Retrieve top-k chunks
    // ---------------------------------------------------------------------------
    static List<String> retrieve(
            EmbeddingStore<TextSegment> store,
            EmbeddingModel embeddingModel,
            String query,
            int topK) {

        Embedding queryEmbedding = embeddingModel.embed(query).content();
        List<EmbeddingMatch<TextSegment>> matches =
                store.findRelevant(queryEmbedding, topK);

        System.out.printf("%n[Retrieve] Query: \"%s\"%n", query);
        System.out.printf("[Retrieve] Top %d results (cosine similarity):%n", topK);

        List<String> chunks = matches.stream()
                .map(match -> {
                    String text = match.embedded().text();
                    double score = match.score();
                    System.out.printf("  sim=%.4f | %s...%n",
                            score, text.substring(0, Math.min(72, text.length())));
                    return text;
                })
                .toList();

        return chunks;
    }

    // ---------------------------------------------------------------------------
    // Stage 3: Assemble RAG prompt
    // ---------------------------------------------------------------------------
    static String buildRagPrompt(String query, List<String> contextChunks) {
        var sb = new StringBuilder();
        sb.append("You are a technical documentation assistant.\n");
        sb.append("Answer using ONLY the provided context.\n");
        sb.append("If the context lacks the answer, say: 'I don't have that information.'\n\n");

        for (int i = 0; i < contextChunks.size(); i++) {
            sb.append(String.format("[Context %d]%n%s%n%n", i + 1, contextChunks.get(i)));
        }

        sb.append(String.format("Question: %s%n%nAnswer:", query));
        return sb.toString();
    }

    // ---------------------------------------------------------------------------
    // Main
    // ---------------------------------------------------------------------------
    public static void main(String[] args) {
        System.out.println("=== RAG Pipeline (Java / LangChain4j) ===\n");

        // Quantized ONNX model — runs on CPU, no GPU required
        EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

        // Stage 1: Indexing
        EmbeddingStore<TextSegment> store = buildVectorStore(embeddingModel, DOCUMENTS);

        // Stage 2: Retrieval
        String query = "What embedding model runs on CPU and what size vectors does it produce?";
        List<String> chunks = retrieve(store, embeddingModel, query, 3);

        // Stage 3: Prompt assembly
        String prompt = buildRagPrompt(query, chunks);
        System.out.println("\n[Prompt] Assembled RAG prompt:");
        prompt.lines().forEach(line -> System.out.println("  " + line));
    }
}
```

**Key notes on the Java implementation:**

- `AllMiniLmL6V2QuantizedEmbeddingModel` bundles the ONNX-quantized model — no external download required at runtime.
- `InMemoryEmbeddingStore` is sufficient for development; swap to `ChromaEmbeddingStore` from `langchain4j-chroma` for production with a running Chroma server.
- `DocumentSplitters.recursive(300, 60)` mirrors the Python chunker: 300-character chunks with 60-character overlap.
- `match.score()` returns cosine similarity directly (not distance), unlike ChromaDB which returns distance.

## Stack Comparison

| Dimension | Python | Java |
|-----------|--------|------|
| Embedding model | `sentence-transformers` — local, full model | `langchain4j-embeddings-all-minilm-l6-v2-q` — bundled ONNX, quantized |
| Vector store (dev) | `chromadb.Client()` in-memory | `InMemoryEmbeddingStore` |
| Vector store (prod) | `chromadb` persistent / Docker | `ChromaEmbeddingStore` via `langchain4j-chroma` |
| RAG orchestration | `LangChain` `RetrievalQA` / `LCEL` chains | `LangChain4j` `AiServices` + `EmbeddingStoreContentRetriever` |
| LLM integration | OpenAI, Anthropic, Ollama, `transformers` | LangChain4j model adapters (OpenAI, Ollama, Bedrock) |
| Spring Boot fit | Not applicable | First-class — `spring-ai` offers drop-in RAG beans |
| Ecosystem maturity | Production-ready, broadest tooling | Rapidly maturing; excellent for JVM backend services |

## Key Takeaways

- **Cosine similarity is the retriever's relevance signal** — it measures the angle between embedding vectors, so semantically related texts score near 1.0 regardless of their surface word overlap.
- **Chunking strategy determines retrieval quality** — overlapping chunks prevent context from being severed at boundaries; tune `chunk_size` and `chunk_overlap` against your documents before tuning the model.
- **RAG decouples knowledge from model weights** — new documents enter the system by updating the vector index, not by retraining; this makes knowledge updates orders of magnitude cheaper than fine-tuning.
