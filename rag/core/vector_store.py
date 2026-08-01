import atexit

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from .embedder import Embedder


class VectorStore:
    def __init__(
        self,
        embedder: Embedder,
        path: str = "./store",
        collection_name: str = "rag",
    ):
        self.client = QdrantClient(path=path)
        atexit.register(self.close)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        if not self.client.collection_exists(collection_name):
            dense_dim = len(embedder.embeddings.embed_query("test"))

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    "langchain-sparse": models.SparseVectorParams()
                },
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedder.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 4) -> list[Document]:
        return self.vector_store.similarity_search(query=query, k=k)

    async def search_async(self, query: str, k: int = 4) -> list[Document]:
        return await self.vector_store.asimilarity_search(query=query, k=k)

    def close(self) -> None:
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
