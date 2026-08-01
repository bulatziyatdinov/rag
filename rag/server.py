from fastapi import FastAPI

from .config import (
    RAG_ASK_QUERY_K_LIMIT,
    RAG_EMBEDDING_MODEL,
    RAG_LLM_MODEL,
    RAG_NUM_CTX,
    RAG_NUM_PREDICT,
    RAG_OLLAMA_BASE_URL,
    RAG_QDRANT_COLLECTION_NAME,
    RAG_QDRANT_PATH,
    RAG_REASONING,
    RAG_SEED,
    RAG_TEMPERATURE,
)
from .core.embedder import Embedder
from .core.vector_store import VectorStore
from .rag import RAG

embedder = Embedder(
    RAG_EMBEDDING_MODEL,
    RAG_OLLAMA_BASE_URL,
)
vector_store = VectorStore(
    embedder,
    RAG_QDRANT_PATH,
    RAG_QDRANT_COLLECTION_NAME,
)

rag = RAG(
    RAG_LLM_MODEL,
    RAG_OLLAMA_BASE_URL,
    RAG_TEMPERATURE,
    RAG_NUM_CTX,
    RAG_NUM_PREDICT,
    RAG_REASONING,
    embedder,
    vector_store,
    RAG_SEED,
)

app = FastAPI()


@app.get("/ask/{query}")
async def ask(query: str):
    response = await rag.ask_async(query, RAG_ASK_QUERY_K_LIMIT)
    return {
        "content": response.content,
        "model": response.response_metadata.get("model", "unknown"),
        "total_duration": response.response_metadata.get("total_duration", "unknown"),
        "usage_metadata": response.usage_metadata,
        "type": "ai",
    }


@app.get("/health")
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy"}