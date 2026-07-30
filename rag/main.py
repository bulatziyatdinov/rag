from .config import (
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


def main():
    try:
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

        while True:
            print("[QUESTION]: ", end="")
            query = input().strip()
            if not query:
                print("[WARNING]: Empty request")

            processed_query = query.lower().lstrip("/")
            if processed_query in {"info", "i"}:
                print("TODO: INFO")
            elif processed_query in {"exit", "e", "quit", "q"}:
                break
            else:
                response = rag.ask(query)
                # TODO: RICH printing or tqdm
                print(f"[ANSWER]: {response.content}")

            print("-" * 40)

    except ConnectionError as ex:
        print(f"[ERROR] No connection with Ollama: {ex}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
