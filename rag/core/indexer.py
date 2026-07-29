import os
import warnings
from urllib.parse import urlparse

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain_community")

import requests
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_excel_loader import StructuredExcelLoader

from .chunker import Chunker
from .embedder import Embedder
from .vector_store import VectorStore


class Indexer:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self,
            data_dir: str = "data",
            data_links: list[str] | None = None,
            save_chunks: bool = False,
            ) -> tuple[int, int]:
        print(f"[INFO] Выбрана модель \"{self.embedder.model}\" для эмбеддинга")

        if data_links is not None:
            self._download_links(data_links, data_dir)

        documents, num_docs = self._load_data(data_dir)

        if not documents:
            return 0, 0

        chunks = self.chunker.split(documents)

        if save_chunks:
            self.chunker.save_chunks(chunks)

        self._index_docs(chunks)

        return num_docs, len(chunks)

    def _download_links(self, data_links: list[str], data_dir: str)-> None:
        num_files = len(data_links)
        for i, url in enumerate(data_links, 1):
            url = url.strip()
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            filepath = os.path.join(data_dir, filename)

            if os.path.exists(filepath):
                print(f'[INFO] Файл {filename} уже существует, пропускаем скачивание')
                continue

            try:
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"[INFO] Скачан {i}/{num_files}: {filename}")
            except requests.RequestException as ex:
                print(
                    f"[WARNING] Ошибка при скачивании {i}/{num_files} \"{url}\": {ex}")

    def _load_data(self, data_dir: str) -> tuple[list[Document], int]:
        documents = []
        num_docs = 0

        for root, _, files in os.walk(data_dir):
            for file in files:
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[-1].lower()

                if ext in {'.txt', '.md'}:
                    loader = TextLoader(filepath, encoding='utf-8')
                elif ext == '.pdf':
                    loader = PyMuPDFLoader(filepath, mode='single')
                elif ext == ".csv":
                    loader = CSVLoader(filepath)
                elif ext in {'.xlsx', 'xls'}:
                    loader = StructuredExcelLoader(filepath)
                else:
                    print(f"[INFO] Индексация пропущена {file}. Неизвестный формат")
                    continue

                loaded_docs = loader.load()

                # NOTE: len(documents) != num_docs in many cases, so we need counter
                num_docs += 1

                documents.extend(loaded_docs)

        return documents, num_docs

    def _index_docs(self, chunks: list[Document]) -> None:
        self.vector_store.add_documents(chunks)
