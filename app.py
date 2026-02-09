from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

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

MODEL_ID = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_ID)

# Store uploaded files in a dictionary: {payroll_area: uploaded_file_object}
uploaded_files: Dict[str, any] = {}
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# LIFESPAN
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global uploaded_files

    # Mapping of payroll areas to their respective PDF filenames
    # In a real scenario, you'd place these files in the directory
    pdf_mapping = {
        "Maharashtra": "maharashtra.pdf",
        "Gujarat": "gujarat.pdf",
        "default": "emp.pdf"
    }

    for area, filename in pdf_mapping.items():
        pdf_path = BASE_DIR / filename
        if pdf_path.exists():
            try:
                uploaded_file = genai.upload_file(str(pdf_path))
                uploaded_files[area.lower()] = uploaded_file
                print(f"✅ PDF uploaded for {area}: {filename}")
            except Exception as e:
                print(f"❌ Failed to upload {filename}: {e}")
        else:
            print(f"⚠️ Warning: {filename} not found at {pdf_path}. Fallback will be used.")

    # Always ensure at least the default 'emp.pdf' is attemptedly loaded
    if "default" not in uploaded_files:
        default_path = BASE_DIR / "emp.pdf"
        if default_path.exists():
            uploaded_files["default"] = genai.upload_file(str(default_path))
            print("✅ Default PDF (emp.pdf) uploaded.")

    yield
    print("🛑 App shutting down")

# --------------------------------------------------
# FASTAPI SETUP
# --------------------------------------------------
app = FastAPI(
    title="HR Chatbot API",
    version="1.1.0",
    lifespan=lifespan
)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    payroll_area: Optional[str] = "default"

# --------------------------------------------------
# ASK ENDPOINT
# --------------------------------------------------
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    area = (request.payroll_area or "default").lower()
    
    # Select the PDF based on payroll area, fallback to default if not found
    pdf_to_use = uploaded_files.get(area) or uploaded_files.get("default")

    if not pdf_to_use:
        raise HTTPException(status_code=503, detail="Policy documents not loaded")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    prompt = f"""
You are an intelligent HR assistant. 
You are assisting an employee from the {area.capitalize()} region.

RULES:
1. Answer strictly based on the provided PDF handbook.
2. If the user asks about state-specific policies, ensure you prioritize the {area.capitalize()} rules found in the document.
3. Reply in English.
4. If the answer is not in the handbook, say you don't have that information for the {area.capitalize()} region.

Question:
{request.question}
"""

    try:
        response = model.generate_content([pdf_to_use, prompt])
        answer = response.text.strip() if response.text else "No response."

        return {
            "answer": answer,
            "response_time": round(time.time() - start_time, 2),
            "region_context": area
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
