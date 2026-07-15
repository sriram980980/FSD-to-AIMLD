// Dependencies (Maven):
// dev.langchain4j:langchain4j:0.31.0
// dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2-q:0.31.0
//
// Node: 4.3 — Augmented Generation — RAG & Vector Search
// Run: javac StarterAssignment.java && java StarterAssignment
//
// Note: The /ingest and /ask HTTP endpoints are Python/FastAPI only.
// This file demonstrates the RAG pipeline (indexing, retrieval, prompt
// assembly) using LangChain4j's in-memory embedding store — equivalent
// to Tasks 1-4 in starter.py.

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

import java.util.ArrayList;
import java.util.List;

public class StarterAssignment {

    // -----------------------------------------------------------------------
    // Sample corpus — mirrors the Python starter's load_sample_corpus()
    // -----------------------------------------------------------------------
    private static List<String> loadSampleCorpus() {
        return List.of(
            "FastAPI is a modern Python web framework built on Starlette and Pydantic. "
                + "It auto-generates OpenAPI docs and supports async request handlers natively. "
                + "Declare endpoints with type-annotated function signatures; FastAPI validates "
                + "all incoming JSON automatically using Pydantic models.",

            "Uvicorn is an ASGI server implementation for Python. It uses uvloop and "
                + "httptools for high-performance HTTP serving. Start a FastAPI app with: "
                + "uvicorn app:app --reload. For production, run behind Nginx as a reverse proxy.",

            "ChromaDB is an open-source vector database optimized for storing embeddings. "
                + "It supports in-memory and persistent modes and performs approximate nearest-"
                + "neighbour search using HNSW. Collections store (id, embedding, document, "
                + "metadata) tuples and expose a query() interface for top-k similarity search.",

            "Sentence Transformers produce dense vector embeddings from text. The model "
                + "all-MiniLM-L6-v2 outputs 384-dimensional float32 vectors and runs on CPU "
                + "with acceptable latency. It is trained via contrastive learning on sentence "
                + "pairs so semantically similar sentences cluster together in vector space.",

            "RAG (Retrieval-Augmented Generation) pairs a retriever with a language model. "
                + "The retriever fetches relevant chunks from a vector store; the language model "
                + "conditions its generation on those chunks. The model never sees the full "
                + "corpus — only the top-k most relevant passages fit inside the context window.",

            "Token limits constrain how much context an LLM can process at once. GPT-4 "
                + "supports 128k tokens; open-weight models like Llama-3-8B typically cap at "
                + "8k tokens. Chunking strategy must keep each chunk well under this limit so "
                + "multiple chunks fit alongside the question and system prompt.",

            "Cosine similarity measures the angle between two vectors regardless of their "
                + "magnitude. Values range from -1 (opposite direction) to +1 (identical "
                + "direction). In retrieval, chunks whose embedding points in the same direction "
                + "as the query embedding are ranked highest by the vector store.",

            "LangChain provides composable primitives for building LLM applications. Its "
                + "LCEL (LangChain Expression Language) lets you chain retrievers, prompts, and "
                + "models with the pipe operator. LangChain4j mirrors these abstractions for JVM "
                + "services, making RAG pipelines portable across Python and Java backends."
        );
    }

    // -----------------------------------------------------------------------
    // Implemented helper — pretty-print retrieval hits
    // -----------------------------------------------------------------------
    private static void printRetrievalResults(List<EmbeddingMatch<TextSegment>> matches) {
        for (int i = 0; i < matches.size(); i++) {
            var match = matches.get(i);
            String preview = match.embedded().text()
                .replace("\n", " ")
                .substring(0, Math.min(72, match.embedded().text().length()));
            System.out.printf("  [%d] sim=%.4f | %s...%n",
                i + 1, match.score(), preview);
        }
    }

    // -----------------------------------------------------------------------
    // Task 1 — student implements this
    // Equivalent to chunk_documents() in starter.py
    // -----------------------------------------------------------------------
    /**
     * Index all documents from {@code rawDocs} into an in-memory embedding store.
     *
     * <p>Steps:
     * <ol>
     *   <li>Create a {@link DocumentSplitters#recursive(int, int)} splitter with
     *       {@code chunkSize=400} and {@code overlap=80}.</li>
     *   <li>For each raw string, wrap it in {@link Document#from(String)}, then
     *       call {@code splitter.splitDocument(document)} to get
     *       {@code List<TextSegment>}.</li>
     *   <li>Embed all segments: {@code embeddingModel.embedAll(segments).content()}.</li>
     *   <li>Add embeddings and segments to {@code store} via
     *       {@code store.addAll(embeddings, segments)}.</li>
     *   <li>Return the total segment count across all documents.</li>
     * </ol>
     */
    static int buildVectorStore(
            EmbeddingModel embeddingModel,
            EmbeddingStore<TextSegment> store,
            List<String> rawDocs) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 2 — student implements this
    // Equivalent to retrieve_top_k() in starter.py
    // -----------------------------------------------------------------------
    /**
     * Retrieve the top-{@code k} most similar chunks for {@code query}.
     *
     * <p>Steps:
     * <ol>
     *   <li>Embed the query string:
     *       {@code embeddingModel.embed(query).content()}.</li>
     *   <li>Call {@code store.findRelevant(queryEmbedding, k)} to get
     *       {@code List<EmbeddingMatch<TextSegment>>}.</li>
     *   <li>Return the list directly — LangChain4j returns cosine
     *       <em>similarity</em> (not distance) via {@code match.score()}.</li>
     * </ol>
     */
    static List<EmbeddingMatch<TextSegment>> retrieveTopK(
            EmbeddingModel embeddingModel,
            EmbeddingStore<TextSegment> store,
            String query,
            int k) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Task 3 — student implements this
    // Equivalent to assemble_rag_prompt() in starter.py
    // -----------------------------------------------------------------------
    /**
     * Build the final RAG prompt string from the query and retrieved chunks.
     *
     * <p>Required structure:
     * <pre>
     * You are a technical documentation assistant.
     * Answer the question using ONLY the provided context.
     * If the context does not contain the answer, reply: 'I don't have that information.'
     *
     * [Context 1]
     * &lt;chunk text&gt;
     *
     * [Context 2]
     * &lt;chunk text&gt;
     *
     * Question: &lt;query&gt;
     *
     * Answer:
     * </pre>
     *
     * Use {@code match.embedded().text()} to extract the chunk text.
     */
    static String assembleRagPrompt(
            String query,
            List<EmbeddingMatch<TextSegment>> contextMatches) {
        // TODO: implement this
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // -----------------------------------------------------------------------
    // Main — calls all three tasks in sequence; prints labeled output
    // -----------------------------------------------------------------------
    public static void main(String[] args) {
        System.out.println("=== RAG Pipeline (Java / LangChain4j) ===\n");

        // Quantized ONNX model — bundled jar, no download needed
        EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        List<String> corpus = loadSampleCorpus();

        // Task 1: Index
        try {
            int totalChunks = buildVectorStore(embeddingModel, store, corpus);
            System.out.printf("[Task 1] buildVectorStore → %d chunks indexed%n", totalChunks);
        } catch (UnsupportedOperationException e) {
            System.out.println("[Task 1] " + e.getMessage() + " — implement buildVectorStore()");
            return;
        }

        // Task 2: Retrieve
        String query = "How does FastAPI handle async requests?";
        List<EmbeddingMatch<TextSegment>> matches;
        try {
            matches = retrieveTopK(embeddingModel, store, query, 3);
            System.out.printf("%n[Task 2] retrieveTopK query: \"%s\"%n", query);
            printRetrievalResults(matches);
        } catch (UnsupportedOperationException e) {
            System.out.println("[Task 2] " + e.getMessage() + " — implement retrieveTopK()");
            return;
        }

        // Task 3: Assemble prompt
        try {
            String prompt = assembleRagPrompt(query, matches);
            System.out.printf("%n[Task 3] assembleRagPrompt → %d char prompt%n", prompt.length());
            System.out.println("\n--- Prompt Preview (first 400 chars) ---");
            System.out.println(prompt.substring(0, Math.min(400, prompt.length()))
                + (prompt.length() > 400 ? "..." : ""));
        } catch (UnsupportedOperationException e) {
            System.out.println("[Task 3] " + e.getMessage() + " — implement assembleRagPrompt()");
            return;
        }

        System.out.println("\n[Done] All three RAG pipeline stages completed.");
        System.out.println("[Note] HTTP endpoints (/ingest, /ask) are in starter.py (FastAPI).");
    }
}
