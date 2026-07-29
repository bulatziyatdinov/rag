import os

from dotenv import load_dotenv

load_dotenv()

RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3.5:2b-q4_K_M")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma:300m")
RAG_OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434")

RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "100"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
