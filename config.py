import os

# Paths
PDF_PATH = "emp.pdf"
CHROMA_DB_PATH = "./chroma_db"

# Model Settings - SPEED OPTIMIZED
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG Settings - OPTIMIZED FOR DISTRIBUTED INFORMATION
# Increased chunk size to preserve more context per chunk
CHUNK_SIZE = 800  # Increased from 500 - keeps policy sections more intact
CHUNK_OVERLAP = 200  # Increased from 100 - ensures continuity across boundaries

# Dynamic retrieval: fetch more chunks, let LLM filter
RETRIEVAL_K_MIN = 3  # Minimum chunks for narrow questions
RETRIEVAL_K_MAX = 8  # Maximum chunks for broad questions
RETRIEVAL_K_DEFAULT = 5  # Default for medium-breadth questions

# Score thresholds for retrieval confidence
SIMILARITY_THRESHOLD_HIGH = 0.7  # Strong single-chunk match
SIMILARITY_THRESHOLD_LOW = 0.4   # Minimum relevance threshold

# LLM Settings
LLM_TEMPERATURE = 0.1  # Low temperature for factual responses
NUM_CTX = 2048  # Increased from 1024 - accommodate more chunks
NUM_PREDICT = 200  # Slightly increased for summarization

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_CONCURRENT_REQUESTS = 3

# Advanced Ollama Performance Settings
OLLAMA_NUM_THREAD = 4
OLLAMA_NUM_GPU = 0