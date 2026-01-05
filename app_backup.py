from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import time
import asyncio
from chatbot import HandbookChatbot
import config

# Global chatbot instance
chatbot = None
semaphore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot, semaphore
    print("🚀 Starting Employee Handbook Chatbot API...")
    print("=" * 50)
    chatbot = HandbookChatbot()
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    success = chatbot.initialize()
    if not success:
        print("❌ Failed to initialize chatbot")
    print("=" * 50)
    
    yield
    
    # Shutdown (cleanup if needed)
    print("Shutting down...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Employee Handbook Chatbot API",
    description="RAG-based chatbot for Navneet Education employee handbook with adaptive retrieval",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class Question(BaseModel):
    question: str
    
class InitRequest(BaseModel):
    force_rebuild: Optional[bool] = False

# Response models
class AnswerResponse(BaseModel):
    question: str
    answer: str
    response_time: float

# Health check
@app.get("/")
async def root():
    return {
        "service": "Employee Handbook Chatbot",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Adaptive retrieval (dynamic k)",
            "Multi-chunk aggregation",
            "Question breadth classification",
            "Distributed information handling"
        ],
        "initialized": chatbot.is_initialized if chatbot else False
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if chatbot and chatbot.is_initialized else "initializing",
        "model": config.LLM_MODEL,
        "ready": chatbot.is_initialized if chatbot else False
    }

# Initialize/rebuild endpoint
@app.post("/initialize")
async def initialize(request: InitRequest):
    global chatbot
    if not chatbot:
        chatbot = HandbookChatbot()
    
    success = chatbot.initialize(force_rebuild=request.force_rebuild)
    
    if success:
        return {"status": "success", "message": "Chatbot initialized successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to initialize chatbot")

# Main query endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(question: Question):
    if not chatbot or not chatbot.is_initialized:
        raise HTTPException(status_code=503, detail="Chatbot not initialized. Please wait or call /initialize")
    
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Limit concurrent requests
    async with semaphore:
        start_time = time.time()
        
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, chatbot.ask, question.question)
        
        response_time = time.time() - start_time
        
        if result.get("success"):
            return {
                "question": question.question,
                "answer": result["answer"],
                "response_time": round(response_time, 2)
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

# Example questions - updated to show breadth variety
@app.get("/examples")
async def get_examples():
    return {
        "examples": {
            "narrow": [
                "What is the probation period?",
                "What are the office timings?",
                "What is the retirement age?"
            ],
            "medium": [
                "How many casual leaves do I get?",
                "How do I apply for maternity leave?",
                "What is the notice period for resignation?"
            ],
            "broad": [
                "What benefits are provided?",
                "What are all the leave policies?",
                "Tell me about compensation",
                "What insurance coverage do we have?"
            ]
        }
    }

# Stats endpoint
@app.get("/stats")
async def get_stats():
    if not chatbot or not chatbot.vectorstore:
        return {"error": "Not initialized"}
    
    return {
        "total_chunks": chatbot.vectorstore._collection.count(),
        "model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "max_concurrent": config.MAX_CONCURRENT_REQUESTS,
        "retrieval_strategy": "adaptive",
        "k_range": f"{config.RETRIEVAL_K_MIN}-{config.RETRIEVAL_K_MAX}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )