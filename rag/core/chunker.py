from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', '.', ' ', ''],
            keep_separator=False,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)

    def save_chunks(self, chunks: list[Document]) -> None:
        with open("chunks.txt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(
                    f"Chunk {i}\n"
                    f"Metadata: {chunk.metadata}\n"
                    + str(chunk.page_content)
                    + f"\n{'-' * 40}\n"
                )
