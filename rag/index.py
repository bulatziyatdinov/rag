from .config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_DATA_DIR,
    RAG_EMBEDDING_MODEL,
    RAG_OLLAMA_BASE_URL,
    RAG_QDRANT_COLLECTION_NAME,
    RAG_QDRANT_PATH,
    RAG_URLS_FILE,
)
from .core.chunker import Chunker
from .core.embedder import Embedder
from .core.indexer import Indexer
from .core.vector_store import VectorStore


def main():
    chunker = Chunker(RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
    embedder = Embedder(RAG_EMBEDDING_MODEL, RAG_OLLAMA_BASE_URL)
    vector_store = VectorStore(embedder, RAG_QDRANT_PATH, RAG_QDRANT_COLLECTION_NAME)

    urls = []

    try:
        with open(RAG_URLS_FILE, "r", encoding="utf-8") as f:
            urls = [url.strip() for url in f.readlines()]
            if urls:
                print(f"[INFO] Файл \"{RAG_URLS_FILE}\" с ссылками на файлы "
                      "для загрузки прочитан")
            else:
                print(f"[INFO] Файл \"{RAG_URLS_FILE}\" с ссылками на файлы пуст")
    except FileNotFoundError:
        print(f"[WARNING] Файл \"{RAG_URLS_FILE}\" с ссылками на файлы не найден")

    indexer = Indexer(chunker, embedder, vector_store)
    num_docs, num_chucks = indexer.run(RAG_DATA_DIR, urls, True)

    print(f"[INFO] Обработано {num_docs} документов. Создано {num_chucks} чанков")


if __name__ == "__main__":
    main()
