# TODO: Create FastAPI HR Chatbot Backend

## Tasks
- [ ] Update requirements.txt to include google-generativeai
- [ ] Create new app.py with FastAPI backend using Gemini 1.5 Flash Lite
- [ ] Implement PDF upload to Gemini on startup
- [ ] Create POST /ask endpoint with question input and answer/response_time output
- [ ] Add CORS middleware
- [ ] Handle cases where answer not in PDF
- [ ] Test the API

## Notes
- Use API key: AIzaSyABeE_WtjD_JOA_gtXhrW-boSKZ6VTQUeU
- Model: Gemini 1.5 Flash Lite
- PDF: emp.pdf
- Endpoint: POST /ask {"question": "string"} -> {"answer": "string", "response_time": float}
