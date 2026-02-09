from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import time
import os
from typing import Dict, Optional
from google import genai

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

# Initialize GenAI Client
# We do this globally, but verify API_KEY in the request
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables!")

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
app = FastAPI(title="HR Chatbot API", version="1.2.0")

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
    global client
    
    # 1. Validation
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing. Please set it in Vercel Environment Variables.")
    
    if not client:
        client = genai.Client(api_key=API_KEY)

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
            raise HTTPException(status_code=503, detail=f"Handbooks missing: {pdf_path} not found")

    # 3. Upload/Retrieve File
    try:
        if filename not in uploaded_files:
            print(f"Uploading {filename} to Gemini...")
            # Use the new SDK for upload
            uploaded_file = client.files.upload(path=str(pdf_path))
            uploaded_files[filename] = uploaded_file
        
        file_node = uploaded_files[filename]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Upload Error: {str(e)}")

    # 4. Generate Content
    start_time = time.time()
    prompt = f"""
You are an intelligent HR assistant. 
You are assisting an employee from the '{area_code.upper()}' Payroll Area.

RULES:
1. Answer strictly based on the provided PDF handbook.
2. If the user asks about policies specific to their Payroll Area ({area_code.upper()}), ensure you prioritize those rules found in the document.
3. Reply in English.
4. If the answer is not in the handbook, say you don't have that specific information for Payroll Area {area_code.upper()}.

Question: {request.question}
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[file_node, prompt]
        )
        
        # In new SDK, response object has different structure
        answer = response.text if response.text else "The AI could not generate a response from the document."
        
        return {
            "answer": answer,
            "response_time": round(time.time() - start_time, 2),
            "region_context": area_code
        }
    except Exception as e:
        # Pass the full error to help debugging
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")
