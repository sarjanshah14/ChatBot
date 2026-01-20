from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import time
import os
from contextlib import asynccontextmanager

import google.generativeai as genai

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)

MODEL_ID = "gemini-2.0-flash"  # Updated to a valid model ID
model = genai.GenerativeModel(MODEL_ID)

uploaded_file = None

# --------------------------------------------------
# LIFESPAN (REPLACES on_event)
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global uploaded_file

    pdf_path = Path("emp.pdf")
    if not pdf_path.exists():
        raise RuntimeError("emp.pdf not found")

    uploaded_file = genai.upload_file(pdf_path)
    print(f"✅ PDF uploaded: {uploaded_file.name}")

    yield  # ---- app runs here ----

    # (Optional cleanup if needed)
    print("🛑 App shutting down")

# --------------------------------------------------
# FASTAPI SETUP
# --------------------------------------------------
app = FastAPI(
    title="HR Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)
@app.get("/")
def home():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

# --------------------------------------------------
# ASK ENDPOINT
# --------------------------------------------------
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not uploaded_file:
        raise HTTPException(status_code=503, detail="PDF not loaded")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    prompt = f"""
You are an HR assistant.

RULES:
- Answer ONLY from the uploaded employee handbook PDF.
- If the answer is not found in the document, say:
  "I only know about company policy information from the handbook."

Question:
{request.question}
"""

    try:
        response = model.generate_content([uploaded_file, prompt])

        answer = response.text.strip() if response.text else "No response."

        return {
            "answer": answer,
            "response_time": round(time.time() - start_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
