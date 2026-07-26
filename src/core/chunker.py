import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int,):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', '.', ' ', ''],
            keep_separator=False,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)

    def save_chunks(self, chunks: list[Document], output_dir: str) -> None:
        chunk_path = os.path.join(output_dir, "chunks.txt")
        with open(chunk_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, 1):
                source_name = chunk.metadata.get("source_name", "unknown")
                f.write(
                    f"Chunk {i}\n"
                    f"Source: {source_name}\n"
                    + str(chunk.page_content)
                    + f"\n{'-' * 40}\n"
                )
