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
    # Currently mapping all N1-N4 codes to the core emp.pdf handbook
    pdf_mapping = {
        "n1": "emp.pdf",
        "n2": "emp.pdf",
        "n3": "emp.pdf",
        "n4": "emp.pdf",
        "maharashtra": "maharashtra.pdf",
        "gujarat": "gujarat.pdf",
        "default": "emp.pdf"
    }

    for area, filename in pdf_mapping.items():
        pdf_path = BASE_DIR / filename
        # Only upload if the file exists and hasn't been uploaded yet (to avoid duplicates for N1-N4)
        if pdf_path.exists() and filename not in [f.display_name for f in uploaded_files.values() if hasattr(f, 'display_name')]:
            try:
                uploaded_file = genai.upload_file(str(pdf_path))
                uploaded_files[area.lower()] = uploaded_file
                print(f"✅ PDF uploaded for {area}: {filename}")
            except Exception as e:
                print(f"❌ Failed to upload {filename}: {e}")
        elif pdf_path.exists():
            # If file already uploaded for another key, reuse the reference
            for existing_area, existing_file in uploaded_files.items():
                # This is a bit of a shortcut, we'll just check if the filename matches our mapping
                if pdf_mapping.get(existing_area) == filename:
                    uploaded_files[area.lower()] = existing_file
                    break

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
    version="1.1.1",
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
    area_code = (request.payroll_area or "default").lower()
    
    # Select the PDF based on payroll area, fallback to default if not found
    pdf_to_use = uploaded_files.get(area_code) or uploaded_files.get("default")

    if not pdf_to_use:
        raise HTTPException(status_code=503, detail="Policy documents not loaded")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    prompt = f"""
You are an intelligent HR assistant. 
You are assisting an employee from the '{area_code.upper()}' Payroll Area/Region.

RULES:
1. Answer strictly based on the provided PDF handbook.
2. If the user asks about policies specific to their Payroll Area ({area_code.upper()}), ensure you prioritize those rules found in the document.
3. Reply in English.
4. If the answer is not in the handbook, say you don't have that specific information for Payroll Area {area_code.upper()}.

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
