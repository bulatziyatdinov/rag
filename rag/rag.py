from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .core.embedder import Embedder
from .core.vector_store import VectorStore
from .prompt import SYSTEM_PROMPT


class RAG:
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        reasoning: bool,
        embedder: Embedder,
        vector_store: VectorStore,
        seed: int = 42,
    ):
        self.model = ChatOllama(
            validate_model_on_init=True,
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            seed=seed,
            reasoning=reasoning,
            # streaming=True,
        )
        self.embedder = embedder
        self.vector_store = vector_store

        augmented_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "КОНТЕКСТ: {context}\n\nВопрос: {question}"),
            ]
        )

        self.chain = augmented_prompt | self.model

    def ask(self, query: str, k: int = 4) -> AIMessage:
        docs = self.vector_store.search(query, k)
        context = '\n\n'.join([f"{doc.metadata}\n{doc.page_content}" for doc in docs])

        response = self.chain.invoke({
            "context": context,
            "question": query,
        })

        return response

    async def ask_async(self, query: str, k: int = 4) -> AIMessage:
        docs = await self.vector_store.search_async(query, k)
        context = '\n\n'.join([f"{doc.metadata}\n{doc.page_content}" for doc in docs])

        response = await self.chain.ainvoke({
            "context": context,
            "question":query,
        })

        return response
