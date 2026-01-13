import os

# Paths
PDF_PATH = "emp.pdf"
CHROMA_DB_PATH = "./chroma_db"

# Model Settings - SPEED & ACCURACY OPTIMIZED
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Advanced Ollama Performance Settings
OLLAMA_NUM_THREAD = os.cpu_count() or 4  # Auto-detect CPU cores
OLLAMA_NUM_GPU = 0  # Set to 1 if GPU available
OLLAMA_KEEP_ALIVE = "5m"  # Keep model loaded for faster responses

# RAG Settings - OPTIMIZED FOR ACCURACY
CHUNK_SIZE = 1000  # Larger chunks = better context preservation
CHUNK_OVERLAP = 250  # 25% overlap prevents context loss at boundaries

# Hybrid Retrieval Strategy - REDUCED for shorter responses
RETRIEVAL_K_MIN = 3   # Reduced from 5 - fewer chunks = shorter answers
RETRIEVAL_K_MAX = 4   # Reduced from 12 - even broad queries stay concise
RETRIEVAL_K_DEFAULT = 4  # Reduced from 7 - default is more focused

# Multi-stage retrieval thresholds
SIMILARITY_THRESHOLD_HIGH = 0.75  # High confidence match
SIMILARITY_THRESHOLD_MEDIUM = 0.55  # Medium confidence
SIMILARITY_THRESHOLD_LOW = 0.35   # Minimum relevance (lowered for recall)

# Re-ranking settings (MMR for diversity)
USE_MMR = True  # Maximum Marginal Relevance for diverse results
MMR_LAMBDA = 0.5  # Balance between relevance and diversity

# LLM Settings
LLM_TEMPERATURE = 0.1  # Low temperature for factual responses (increased slightly for better flow)
NUM_CTX = 2048  # Context window (reduced - we don't need huge context)
NUM_PREDICT = 100  # Shorter responses - maximum 150 tokens (was 300)
TOP_P = 0.9  # Nucleus sampling for better quality
REPEAT_PENALTY = 1.1  # Reduce repetition

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_CONCURRENT_REQUESTS = 5  # Increased for better throughput

# Caching settings
ENABLE_RESPONSE_CACHE = True
CACHE_MAX_SIZE = 100  # Cache 100 most recent Q&A pairs
CACHE_TTL = 3600  # 1 hour cache lifetime

# Batch processing
BATCH_EMBEDDING_SIZE = 32  # Process embeddings in batches

# Logging
LOG_LEVEL = "INFO"
LOG_RETRIEVAL_DETAILS = True  # Log retrieval metrics for tuning