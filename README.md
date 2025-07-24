# Product-Query Bot

**Video demo:** [https://youtu.be/rZiE545tC80](https://youtu.be/rZiE545tC80)

A Retrieval-Augmented Generation (RAG) chatbot for answering product-related queries using FastAPI, Gemini (Google Generative AI), and FAISS vector search.

## Features

* Answers user questions about products using context from a product catalog.
* Uses semantic search (FAISS + HuggingFace embeddings) to retrieve relevant product information.
* Generates answers with Gemini (Google Generative AI).
* Maintains user conversation history for context-aware responses.

## Architecture

* **FastAPI**: REST API for user interaction.
* **RAG Pipeline**: Combines retrieval (FAISS) and generation (Gemini) in a graph-based flow.
* **FAISS**: Vector store for fast semantic search over product descriptions.
* **HuggingFace Embeddings**: For encoding product texts and queries.
* **Gemini**: Large language model for answer generation.
* **LangGraph**: Orchestrates the flow between retriever, memory, and responder agents.

## Setup

1. **Clone the repository**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Environment variables**
   Create a `.env` file in the root directory with:

```
GEMINI_API_KEY=your_gemini_api_key_here
TOP_K=3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
PYTHONPATH = "."
```

## Environment Variable Configuration

1. Copy the `.env.example` file to `.env`:

   ```bash
   cp .env.example .env
   ```
2. Edit the `.env` file and add your actual values (e.g., your Gemini API key).
3. You can now run the application locally or with Docker.

## Running the API

1. **Indexing Documents**

   * On startup, the API automatically loads and indexes the product catalog from `docs/products.json`.
   * If you update `docs/products.json`, simply restart the API to re-index the new products.
   * The vector index is stored in the `vectorstore/` directory.

2. **Start the API server**

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Usage

Send a POST request to `/query` with JSON body:

```json
{
  "user_id": "user1",
  "query": "Tell me about Zubale Crunch"
}
```

Response:

```json
{
  "user_id": "user1",
  "answer": "Cereal Zubale Crunch: Cereal de avena integral..."
}
```

## Testing the Flow

1. **Run the test suite**

```bash
pytest
```

2. **What is tested?**

   * The test flow in `app/tests/test_rag.py` covers:

     * Direct pipeline usage: Indexing a product and querying it.
     * API endpoint: Sending a POST request to `/query` and checking the response.

3. **Extending tests**

   * You can add more test cases to `app/tests/test_rag.py` to cover additional scenarios or edge cases.

## Re-indexing Products

* If you change the product catalog (`docs/products.json`), restart the API to re-index.
* To clear the vector index, delete the `vectorstore/` directory before restarting.

## Docker Setup

### **Build the Docker image**

```bash
docker build -t product-query-bot .
```

### **Run the container**

```bash
docker run -p 8000:8000 --env-file .env product-query-bot
```

### **Using docker-compose**

A `docker-compose.yml` file is included for convenience:

```bash
docker compose up --build
```

The API will be accessible at:

```
http://localhost:8000/docs
```

## License

MIT
