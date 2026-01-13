from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict
from contextlib import asynccontextmanager
import time
import asyncio
import logging
from chatbot import HandbookChatbot
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
chatbot = None
semaphore = None
request_times = []  # Track response times for monitoring


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle"""
    global chatbot, semaphore
    
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Starting Employee Handbook Chatbot API v2.0")
    logger.info("=" * 60)
    
    try:
        chatbot = HandbookChatbot()
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
        success = chatbot.initialize()
        if not success:
            logger.error("❌ Failed to initialize chatbot")
            raise RuntimeError("Chatbot initialization failed")
        
        logger.info("✅ API ready to accept requests")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down gracefully...")
    # Add cleanup if needed (e.g., close connections)


# Initialize FastAPI
app = FastAPI(
    title="Employee Handbook Chatbot API",
    description="Production-ready RAG chatbot with adaptive retrieval and caching",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question to ask")
    include_metadata: Optional[bool] = Field(False, description="Include retrieval metadata")


class InitRequest(BaseModel):
    force_rebuild: Optional[bool] = Field(False, description="Force rebuild vector store")


class AnswerResponse(BaseModel):
    question: str
    answer: str
    response_time: float
    cached: Optional[bool] = False
    metadata: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    ready: bool
    cache_enabled: bool
    uptime: Optional[float] = None


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


# Routes
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "service": "Employee Handbook Chatbot",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Adaptive retrieval (dynamic k-selection)",
            "Maximum Marginal Relevance (MMR) for diversity",
            "Response caching for speed",
            "Multi-stage confidence scoring",
            "Optimized chunk processing",
            "Concurrent request handling"
        ],
        "initialized": chatbot.is_initialized if chatbot else False,
        "endpoints": {
            "chat": "/ask",
            "health": "/health",
            "stats": "/stats",
            "examples": "/examples",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if (chatbot and chatbot.is_initialized) else "initializing",
        "model": config.LLM_MODEL,
        "ready": chatbot.is_initialized if chatbot else False,
        "cache_enabled": config.ENABLE_RESPONSE_CACHE
    }


@app.post("/initialize")
async def initialize(request: InitRequest):
    """Initialize or rebuild the chatbot"""
    global chatbot
    
    if not chatbot:
        chatbot = HandbookChatbot()
    
    try:
        success = chatbot.initialize(force_rebuild=request.force_rebuild)
        
        if success:
            logger.info("✓ Chatbot reinitialized")
            return {
                "status": "success",
                "message": "Chatbot initialized successfully",
                "force_rebuild": request.force_rebuild
            }
        else:
            raise HTTPException(status_code=500, detail="Initialization failed")
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(question: Question):
    """
    Main chat endpoint with optimized processing.
    Handles concurrent requests with semaphore.
    """
    # Validation
    if not chatbot or not chatbot.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized. Please wait or call /initialize"
        )
    
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Rate limiting with semaphore
    async with semaphore:
        start_time = time.time()
        
        try:
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                chatbot.ask,
                question.question
            )
            
            response_time = time.time() - start_time
            
            # Track metrics
            request_times.append(response_time)
            if len(request_times) > 100:  # Keep last 100
                request_times.pop(0)
            
            if result.get("success"):
                response = {
                    "question": question.question,
                    "answer": result["answer"],
                    "response_time": round(response_time, 2),
                    "cached": result.get("cached", False)
                }
                
                # Include metadata if requested
                if question.include_metadata and "metadata" in result:
                    response["metadata"] = result["metadata"]
                
                logger.info(f"✓ Answered in {response_time:.2f}s (cached: {result.get('cached', False)})")
                return response
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples")
async def get_examples():
    """Get example questions by complexity"""
    return {
        "examples": {
            "narrow": [
                "What is the probation period?",
                "What are the office timings?",
                "What is the retirement age?",
                "How many casual leaves per year?"
            ],
            "medium": [
                "How do I apply for maternity leave?",
                "What is the notice period for resignation?",
                "What are the working hours?",
                "When can I avail earned leave?"
            ],
            "broad": [
                "What are all the leave policies?",
                "Tell me about employee benefits",
                "What insurance coverage is provided?",
                "Explain the compensation structure",
                "What are the different types of leaves?"
            ]
        },
        "tips": [
            "Be specific for faster, more accurate answers",
            "Use natural language - no special formatting needed",
            "Broad questions may take slightly longer but provide comprehensive answers"
        ]
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics and performance metrics"""
    if not chatbot or not chatbot.vectorstore:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        total_chunks = chatbot.vectorstore._collection.count()
        
        # Calculate metrics
        avg_response_time = sum(request_times) / len(request_times) if request_times else 0
        
        cache_info = {}
        if chatbot.response_cache:
            cache_info = {
                "enabled": True,
                "size": len(chatbot.response_cache.cache),
                "capacity": chatbot.response_cache.capacity,
                "hit_rate": "tracked_in_logs"
            }
        
        return {
            "vectorstore": {
                "total_chunks": total_chunks,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP
            },
            "models": {
                "llm": config.LLM_MODEL,
                "embedding": config.EMBEDDING_MODEL
            },
            "retrieval": {
                "strategy": "adaptive_mmr",
                "k_range": f"{config.RETRIEVAL_K_MIN}-{config.RETRIEVAL_K_MAX}",
                "use_mmr": config.USE_MMR,
                "mmr_lambda": config.MMR_LAMBDA if config.USE_MMR else None
            },
            "performance": {
                "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
                "avg_response_time": round(avg_response_time, 2),
                "total_requests_tracked": len(request_times)
            },
            "cache": cache_info,
            "llm_config": {
                "temperature": config.LLM_TEMPERATURE,
                "context_window": config.NUM_CTX,
                "max_tokens": config.NUM_PREDICT
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/clear")
async def clear_cache():
    """Clear the response cache"""
    if not chatbot or not chatbot.response_cache:
        return {"message": "Cache not enabled or chatbot not initialized"}
    
    chatbot.response_cache.cache.clear()
    logger.info("Cache cleared")
    return {"message": "Cache cleared successfully"}


# Development server
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting development server...")
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )