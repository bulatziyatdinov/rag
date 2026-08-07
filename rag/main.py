import time

from rich import print as rprint
from rich.markdown import Markdown

from .config import (
    RAG_ASK_QUERY_K_LIMIT,
    RAG_DATA_DIR,
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

        info_commands = ("Commands:\n"
                "- info, i - information\n"
                "- exit, e, quit, q - exit")

        info = (f"[*] RAG System [*]\n"
                f"LLM: {RAG_LLM_MODEL}\n"
                f"Embeddings: {RAG_EMBEDDING_MODEL}\n"
                f"Data folder: {RAG_DATA_DIR}\n"
                f"Indexed data (Qdrant): {RAG_QDRANT_PATH}\n"
                f"All settings store in .env file\n\n"
                + info_commands)
        print(info)

        while True:
            print("[QUESTION]: ", end="")
            query = input().strip()
            if not query:
                print("[WARNING]: Empty request")
                continue

            processed_query = query.lower().lstrip("/")
            if processed_query in {"info", "i"}:
                print(info)
            elif processed_query in {"exit", "e", "quit", "q"}:
                break
            else:
                start_time = time.time()
                response = rag.ask(query, RAG_ASK_QUERY_K_LIMIT)
                end_time = time.time() - start_time

                tokens_info = [i[1] for i in response.usage_metadata.items()]
                rprint(
                    Markdown(
                        f"[ANSWER]: {response.content}  \n"
                        f"_Time: {end_time:.3f}s. "
                        f"Tokens Input: {tokens_info[0]}, Output: {tokens_info[1]}, "
                        f"Total: {tokens_info[2]}_"
                    )
                )

            print("-" * 40)

    except ConnectionError as ex:
        print(f"[ERROR] No connection with Ollama: {ex}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
