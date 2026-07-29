from .config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_EMBEDDING_MODEL,
    RAG_OLLAMA_BASE_URL,
)
from .core.chunker import Chunker
from .core.embedder import Embedder
from .core.indexer import Indexer
from .core.vector_store import VectorStore


def main():
    chunker = Chunker(RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
    embedder = Embedder(RAG_EMBEDDING_MODEL, RAG_OLLAMA_BASE_URL)
    vector_store = VectorStore(embedder)

    indexer = Indexer(chunker, embedder, vector_store)
    indexer.run(save_chunks=True)


if __name__ == "__main__":
    main()

