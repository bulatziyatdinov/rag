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


# TODO: this class needs logging
class Indexer:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def run(
            self,
            data_dir: str = "data",
            data_urls: list[str] | None = None,
            save_chunks: bool = False,
            ) -> tuple[int, int]:
        if data_urls:
            num_files = self._download_links(data_urls, data_dir)
            print(f"[INFO] Скачано {num_files} файлов из {len(data_urls)}")
        else:
            print("[INFO] Скачивание файлов пропущено")

        documents, num_docs = self._load_data(data_dir)
        print(f"[INFO] Загружено файлов: {num_docs}")

        if not documents:
            return 0, 0

        chunks = self.chunker.split(documents)

        if save_chunks:
            self.chunker.save_chunks(chunks)

        self._index_docs(chunks)

        return num_docs, len(chunks)

    def _download_links(self, data_urls: list[str], data_dir: str) -> int:
        print("[INFO] Скачивание файлов...")
        num_urls = len(data_urls)
        num_files = 0

        for i, url in enumerate(data_urls, 1):
            url = url.strip()
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            filepath = os.path.join(data_dir, filename)

            file_info = f"[INFO] Файл {i}/{num_urls} \"{filename}\" - "

            if os.path.exists(filepath):
                print(file_info + "Пропущен, уже существует")
                continue

            try:
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                num_files += 1
                print(file_info + "Скачан")
            except requests.RequestException as ex:
                print(
                    f"[WARNING] Файл {i}/{num_urls} по ссылке \"{url}\" не скачан. "
                    f"Ошибка при скачивании: {ex}")

        return num_files

    def _load_data(self, data_dir: str) -> tuple[list[Document], int]:
        print("[INFO] Загрузка файлов для индексации...")
        documents = []
        num_docs = 0

        for root, _, files in os.walk(data_dir):
            num_files = len(files)
            for i, file in enumerate(files, 1):
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[-1].lower()

                file_info = f"[INFO] Файл {i}/{num_files} \"{file}\" - "

                if ext in {".txt", ".md"}:
                    loader = TextLoader(filepath, encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyMuPDFLoader(filepath, mode="single")
                elif ext == ".csv":
                    loader = CSVLoader(filepath)
                elif ext in {".xlsx", "xls"}:
                    loader = StructuredExcelLoader(filepath)
                else:
                    print(file_info + f"Пропущен, неизвестный формат [{ext}]")
                    continue

                loaded_docs = loader.load()

                for doc in loaded_docs:
                    doc.metadata['filename'] = file

                # NOTE: len(documents) != num_docs in many cases, so we need counter
                num_docs += 1
                print(file_info + "Загружен")

                documents.extend(loaded_docs)

        return documents, num_docs

    def _index_docs(self, chunks: list[Document]) -> None:
        print("[INFO] Индексирование файлов...")
        self.vector_store.add_documents(chunks)
