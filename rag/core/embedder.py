from langchain_ollama import OllamaEmbeddings


class Embedder:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.embeddings = OllamaEmbeddings(
            model=model,
            base_url=base_url,
            validate_model_on_init=True,
        )
