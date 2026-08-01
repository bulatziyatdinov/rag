import os

from dotenv import load_dotenv

load_dotenv()

# Data
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
RAG_URLS_FILE = os.getenv("RAG_URLS_FILE", "urls.txt")

# QDRANT
RAG_QDRANT_PATH = os.getenv("RAG_QDRANT_PATH", ".store")
RAG_QDRANT_COLLECTION_NAME = os.getenv("RAG_QDRANT_COLLECTION_NAME", "rag")

# Ollama URL
RAG_OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434")

# LLM parameters
RAG_LLM_MODEL = os.getenv(
    "RAG_LLM_MODEL", "hf.co/techwithsergiu/Qwen3.5-text-2B-GGUF:Q4_K_M")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma:300m")
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
RAG_NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))
RAG_NUM_PREDICT = int(os.getenv("RAG_NUM_PREDICT", "512"))
RAG_REASONING = os.getenv("RAG_REASONING", "false").lower() in {'true', 't', '1'}
RAG_SEED = int(os.getenv("RAG_SEED", "42"))
RAG_ASK_QUERY_K_LIMIT = int(os.getenv("RAG_ASK_QUERY_K_LIMIT", "4"))

# Chunking settings
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
RAG_SAVE_CHUNKS = os.getenv("RAG_SAVE_CHUNKS", "false").lower() in {'true', 't', '1'}
