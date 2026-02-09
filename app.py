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
# CONFIG & CLIENT SETUP
# --------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR = Path(__file__).resolve().parent

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables!")

# Global model instance
model = genai.GenerativeModel("gemini-2.0-flash")

# Cache for uploaded files
uploaded_files: Dict[str, any] = {}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_pdf_for_area(area_code: str):
    """Returns the correct filename for the payroll area."""
    area = area_code.lower()
    mapping = {
        "n1": "emp.pdf",
        "n2": "emp.pdf",
        "n3": "emp.pdf",
        "n4": "emp.pdf",
        "maharashtra": "maharashtra.pdf",
        "gujarat": "gujarat.pdf"
    }
    return mapping.get(area, "emp.pdf")

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(title="HR Chatbot API", version="1.2.2")

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

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # 1. Validation
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing. Please set it in Vercel Environment Variables.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 2. File Handling
    area_code = (request.payroll_area or "default").lower()
    filename = get_pdf_for_area(area_code)
    pdf_path = BASE_DIR / filename
    
    if not pdf_path.exists():
        # Fallback to emp.pdf
        filename = "emp.pdf"
        pdf_path = BASE_DIR / filename
        if not pdf_path.exists():
            raise HTTPException(status_code=503, detail=f"Handbook missing: {filename}")

    # 3. Upload/Retrieve File
    try:
        if filename not in uploaded_files:
            print(f"Uploading {filename} to Gemini...")
            # Use the stable old SDK method
            uploaded_file = genai.upload_file(str(pdf_path))
            uploaded_files[filename] = uploaded_file
        
        file_ref = uploaded_files[filename]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Upload Error (Old SDK): {str(e)}")

    # 4. Generate Content
    start_time = time.time()
    prompt = f"""
You are an intelligent HR assistant. 
The employee is currently mapped to Payroll Area: '{area_code.upper()}'.

RULES:
1. Answer based on the provided PDF handbook.
2. If a policy is general (applies to everyone), provide that information.
3. If the handbook has specific sections for different regions or payroll areas, prioritize the rules for '{area_code.upper()}'.
4. If the handbook doesn't mention '{area_code.upper()}' specifically, use the default or general company policies found in the document.
5. If the information is truly missing from the entire handbook, only then say you don't have that information.
6. Reply in English.

Question: {request.question}
"""

    try:
        response = model.generate_content([file_ref, prompt])
        answer = response.text if response.text else "The AI could not generate a response."
        
        return {
            "answer": answer,
            "response_time": round(time.time() - start_time, 2),
            "region_context": area_code
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")
