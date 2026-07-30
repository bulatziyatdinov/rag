from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from .core.embedder import Embedder
from .core.vector_store import VectorStore


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

    def ask(self, query: str):
        docs = self.vector_store.search(query, 2)
        context = '\n'.join([doc.page_content for doc in docs])
        # TODO: change prompt
        augmented_prompt=(("Ты ассистент для ответов на вопросы. Отвечать необходимо "
                "исключительно по контексту. Дай ответ только на вопрос пользователя "
                "без лишних предварительных ответу слов") + f"\nКонтекст: {context}\n"
                + f"\nВопрос пользователя: {query}")

        response = self.model.invoke(
            [HumanMessage(content=augmented_prompt)]
        )

        return response
