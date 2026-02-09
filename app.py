from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import time
import os
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
    # Important: In Vercel, if this is missing, the app will crash.
    # We allow it to start but check during request to give better error.
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is not set!")

genai.configure(api_key=API_KEY or "DUMMY_KEY")

MODEL_ID = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_ID)

# Store uploaded files in a dictionary: {filename: uploaded_file_object}
uploaded_files: Dict[str, any] = {}
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
async def get_pdf_for_area(area_code: str):
    """Retrieves or uploads the PDF for a specific area code."""
    area = area_code.lower()
    
    # Mapping of codes to files
    pdf_mapping = {
        "n1": "emp.pdf",
        "n2": "emp.pdf",
        "n3": "emp.pdf",
        "n4": "emp.pdf",
        "maharashtra": "maharashtra.pdf",
        "gujarat": "gujarat.pdf",
        "default": "emp.pdf"
    }
    
    filename = pdf_mapping.get(area, "emp.pdf")
    pdf_path = BASE_DIR / filename
    
    if not pdf_path.exists():
        # Fallback to emp.pdf if specific file missing
        pdf_path = BASE_DIR / "emp.pdf"
        if not pdf_path.exists():
            return None

    # Check if we already uploaded this file in this instance session
    if filename in uploaded_files:
        return uploaded_files[filename]

    # Upload to Gemini (Gemini keeps files for 48 hours)
    try:
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY is missing in Vercel settings")
            
        print(f"Uploading {filename} to Gemini...")
        uploaded_file = genai.upload_file(str(pdf_path))
        uploaded_files[filename] = uploaded_file
        return uploaded_file
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

# --------------------------------------------------
# FASTAPI SETUP
# --------------------------------------------------
app = FastAPI(
    title="HR Chatbot API",
    version="1.1.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    # Serve index.html as the main entry point
    return FileResponse(BASE_DIR / "index.html")

class QuestionRequest(BaseModel):
    question: str
    payroll_area: Optional[str] = "default"

# --------------------------------------------------
# ASK ENDPOINT
# --------------------------------------------------
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not API_KEY:
         raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server")

    area_code = (request.payroll_area or "default").lower()
    
    # Get PDF on demand
    pdf_to_use = await get_pdf_for_area(area_code)

    if not pdf_to_use:
        raise HTTPException(status_code=503, detail="Policy documents (PDF) could not be loaded")

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
            "region_context": area_code
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
